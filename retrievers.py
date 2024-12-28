from tqdm import tqdm
import pandas as pd
import pyreadstat
from rank_bm25 import BM25Okapi
import pickle
import numpy as np
import os
import torch
from transformers import T5ForSequenceClassification, T5Tokenizer, AutoTokenizer, AutoModel

# Assuming these functions are provided as described
categories = [
    'F_METRO', 'F_CREGION', 'F_CDIVISION', 'F_USR_SELFID', 'F_AGECAT', 'F_GENDER',
    'F_EDUCCAT', 'F_EDUCCAT2', 'F_HISP', 'F_HISP_ORIGIN', 'F_YEARSINUS_RECODE',
    'F_RACECMB', 'F_RACETHNMOD', 'F_CITIZEN', 'F_BIRTHPLACE', 'F_MARITAL', 'F_RELIG',
    'F_BORN', 'F_RELIGCAT1', 'F_ATTEND', 'F_PARTY_FINAL', 'F_PARTYLN_FINAL',
    'F_PARTYSUM_FINAL', 'F_PARTYSUMIDEO_FINAL', 'F_VOTED2020', 'F_VOTEGEN2020',
    'F_INC_SDT1', 'F_IDEO', 'F_INTFREQ', 'F_VOLSUM', 'F_INC_TIER2', 'POL1JBSTR_W116']

def get_question_and_answers(row, question_code, question_mapping, response_mapping):
    question = question_mapping.get(question_code, "Question code not found")
    answers = response_mapping.get(question_code, "No answers available for this question code")
    actual_answer = row[question_code] if question_code in row else "No answer provided"
    actual_answer_text = answers.get(actual_answer, actual_answer) if isinstance(answers, dict) else actual_answer

    answers = {k: v for k, v in answers.items() if v != 'Refused' or v != 'DK/Refused/No lean'}
    return question, answers, actual_answer_text

def create_user_metadata(row, question_mapping, response_mapping):
    sentence_parts = []
    for category in categories:
        question, _, answer = get_question_and_answers(row, category, question_mapping, response_mapping)
        if pd.isna(answer):
            continue
        sentence_parts.append(f"{question}: {answer}")
    return ", ".join(sentence_parts)

class T5Retriever:
    def __init__(self, model_name="./CRAG/finetuned_t5_evaluator", data_path="/data/matmang/ATP W117.sav", device='cuda', top_n=5):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForSequenceClassification.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.data_path = data_path
        self.top_n = top_n
        self.df, self.meta = self._load_data()
        self.question_mapping = self.meta.column_names_to_labels
        self.response_mapping = self.meta.variable_value_labels
        self.columns_to_drop = [
            'QKEY', 'INTERVIEW_START_W117', 'INTERVIEW_END_W117',
            'WEIGHT_W117', 'WEIGHT_W117_VOTE', 'LANG_W117',
            'FORM_W117', 'DEVICE_TYPE_W117', 'XTABLET_W117', 'F_METRO',
            'F_CREGION', 'F_CDIVISION', 'F_USR_SELFID', 'F_AGECAT',
            'F_GENDER', 'F_EDUCCAT', 'F_EDUCCAT2', 'F_HISP',
            'F_HISP_ORIGIN', 'F_YEARSINUS_RECODE', 'F_RACECMB',
            'F_RACETHNMOD', 'F_CITIZEN', 'F_BIRTHPLACE', 'F_MARITAL',
            'F_RELIG', 'F_BORN', 'F_RELIGCAT1', 'F_RELTRAD', 'F_ATTEND',
            'F_PARTY_FINAL', 'F_PARTYLN_FINAL', 'F_PARTYSUM_FINAL',
            'F_PARTYSUMIDEO_FINAL', 'F_REG', 'F_INC_SDT1', 'F_IDEO',
            'F_INTFREQ', 'F_VOLSUM', 'F_INC_TIER2'
        ]
        self._preprocess_data()

    def _load_data(self):
        df, meta = pyreadstat.read_sav(self.data_path)
        return df, meta

    def _preprocess_data(self):
        self.df = self.df.drop(columns=self.columns_to_drop)

    def _get_question_and_answers_excluding_target(self, row, query_code):
        responses = []
        for question_code in row.index:
            if question_code == query_code:
                continue
            question = self.question_mapping.get(question_code, "Question code not found")
            # question_code와 일치하는 부분 제거
            if question_code in question:
                question = question.replace(question_code + ". ", "")
            answers = self.response_mapping.get(question_code, {})
            actual_answer = row[question_code]
            if pd.isna(actual_answer):
                continue
            actual_answer_text = answers.get(actual_answer, actual_answer) if isinstance(answers, dict) else actual_answer
            responses.append((question, actual_answer_text))
        return responses

    def select_relevants(self, responses, query, top_n=None):
        top_n = self.top_n
        max_length = 512
        responses_data = []
        for i, r in enumerate(responses):
            input_content = query + " [SEP] " + f'Q : {r[0]} A : {r[1]}'
            inputs = self.tokenizer(input_content, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
            try:
                with torch.no_grad():
                    outputs = self.model(inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device))
                scores = float(outputs["logits"].cpu())
            except:
                scores = -1.0
            responses_data.append((scores, r, i))
        sorted_results = sorted(responses_data, key=lambda x: x[0], reverse=True)
        ctxs = [f'Q : {s[1][0]} A : {s[1][1]}' for s in sorted_results[:top_n]]
        idxs = [str(s[2]) for s in sorted_results]
        return ctxs, idxs

    def response_refinement(self, query, user, query_code):
        top_n = self.top_n

        rsps = self._get_question_and_answers_excluding_target(user, query_code)
        results, idxs = self.select_relevants(responses=rsps, query=query, top_n=top_n)

        return results

class BGERetriever:
    def __init__(self, model_name="BAAI/bge-m3", data_path="/data/matmang/ATP W117.sav", device='cuda', top_n=5, query_code='POL1JB_W116', target_indices=None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.data_path = data_path
        self.embedding_path = f'user_embeddings_{query_code}.pkl'
        self.top_n = top_n
        self.query_code = query_code
        self.df, self.meta = self._load_data()
        self.question_mapping = self.meta.column_names_to_labels
        self.response_mapping = self.meta.variable_value_labels
        self.full_df = self.df.copy()
        self.columns_to_drop = [
            'QKEY', 'INTERVIEW_START_W116', 'INTERVIEW_END_W116',
            'WEIGHT_W116', 'XW91NONRESP_W116', 'LANG_W116',
            'FORM_W116', 'DEVICE_TYPE_W116', 'XW78NONRESP_W116',
        ]
        self._preprocess_data(target_indices)
        # self.user_embeddings = self._load_or_create_embeddings()
        # self.user_response_embeddings = self._load_or_create_embeddings()

    def _load_data(self):
        df, meta = pyreadstat.read_sav(self.data_path)
        return df, meta

    def _preprocess_data(self, target_indices=None):
        if target_indices:
            self.df = self.df.iloc[target_indices]
        self.df = self.df.drop(columns=self.columns_to_drop)
        self.df = self.df.dropna(subset=[self.query_code])
        self.df = self.df[self.df[self.query_code] != 99.0]
        self.df = self.df.reset_index(drop=True)
        self.response_mapping['CONG_W116'][2.0] = 'Democratic candidate'
        self.response_mapping['CONG_W116'][1.0] = 'Republican candidate'
        self.response_mapping['CONGA_W116'][2.0] = 'Democratic candidate'
        self.response_mapping['CONGA_W116'][1.0] = 'Republican candidate'


    def _load_or_create_embeddings(self):
        if os.path.exists(self.embedding_path):
            with open(self.embedding_path, 'rb') as f:
                user_embeddings = pickle.load(f)
        else:
            user_embeddings = self._create_embeddings(response_mode=True) # 이거 중요.
            with open(self.embedding_path, 'wb') as f:
                pickle.dump(user_embeddings, f)
        return user_embeddings

    def _create_embeddings(self, response_mode=False):
        user_embeddings = []
        if response_mode:
            with open(os.path.join(f'cache/useful_5_shot_people_100', f'useful_qna_{self.query_code}.pkl'), 'rb') as f:
                user_responses = pickle.load(f)
            for i, row in tqdm(enumerate(self.df.iterrows()), total=len(self.df), desc="Creating embeddings"):
                user_response = user_responses[i]
                inputs = self.tokenizer(user_response, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    user_embedding = self.model(**inputs).last_hidden_state[:, 0, :].detach().cpu()
                user_embeddings.append(user_embedding)
            return user_embeddings
        else:
            for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Creating embeddings"):
                user_metadata = create_user_metadata(row, self.question_mapping, self.response_mapping)
                inputs = self.tokenizer(user_metadata, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    user_embedding = self.model(**inputs).last_hidden_state[:, 0, :].detach().cpu()
                user_embeddings.append(user_embedding)
            return user_embeddings

    def _get_question_and_answers_excluding_target(self, row, query_code, visited):
        responses = []
        for question_code in row.index:
            if question_code == query_code:
                continue
            if question_code in categories:
                continue
            if question_code in visited:
                continue
            question = self.question_mapping.get(question_code, "Question code not found")
            # question_code와 일치하는 부분 제거
            if question_code in question:
                question = question.replace(question_code + ". ", "")
            answers = self.response_mapping.get(question_code, {})
            actual_answer = row[question_code]
            if pd.isna(actual_answer):
                continue
            actual_answer_text = answers.get(actual_answer, actual_answer) if isinstance(answers, dict) else actual_answer
            responses.append((question, actual_answer_text, question_code))
        return responses

    def _get_question_and_answers_useful(self, row, query_code, useful_codes):
        responses = []
        for question_code in useful_codes:
            if question_code == query_code:
                continue
            if question_code in categories:
                continue
            question = self.question_mapping.get(question_code, "Question code not found")
            # question_code와 일치하는 부분 제거
            if question_code in question:
                question = question.replace(question_code + ". ", "")
            answers = self.response_mapping.get(question_code, {})
            actual_answer = row[question_code]
            if pd.isna(actual_answer):
                continue
            actual_answer_text = answers.get(actual_answer, actual_answer) if isinstance(answers, dict) else actual_answer
            responses.append((question, actual_answer_text, question_code))
        return responses

    def select_relevants(self, responses, query, top_n=None):
        if top_n is None:
            top_n = self.top_n
        max_length = 512

        query_inputs = self.tokenizer(query, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(self.device)
        with torch.no_grad():
            query_embedding = self.model(**query_inputs).last_hidden_state[:, 0, :].detach().cpu()

        responses_data = []
        for i, r in enumerate(responses):
            input_content = f'Q : {r[0]} A : {r[1]}'
            inputs = self.tokenizer(input_content, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(self.device)
            try:
                with torch.no_grad():
                    response_embedding = self.model(**inputs).last_hidden_state[:, 0, :].detach().cpu()
                similarity = torch.nn.functional.cosine_similarity(query_embedding, response_embedding).item()
            except Exception as e:
                print(f"Error processing response {i}: {e}")
                similarity = -1.0
            responses_data.append((similarity, r, i))

        sorted_results = sorted(responses_data, key=lambda x: x[0], reverse=True)
        top_responses = sorted_results[:top_n]

        ctxs = [f'Q : {s[1][0]} A : {s[1][1]}' for s in top_responses]
        idxs = [str(s[1][2]) for s in top_responses]

        return ctxs, idxs

    def response_refinement(self, query, user, query_code, useful_codes=None, visited=[], top_n=10):

        if useful_codes is not None:
            rsps = self._get_question_and_answers_useful(user, query_code, useful_codes)
            ctxs = [f'Q : {s[0]} A : {s[1]}' for s in rsps]
            return ctxs
        else:
            rsps = self._get_question_and_answers_excluding_target(user, query_code, visited)

        results, idxs = self.select_relevants(responses=rsps, query=query, top_n=top_n)

        return results, idxs

    def find_similar_users(self, target_user_idx, top_k=5):
        target_user_row = self.df.iloc[target_user_idx]
        target_user_metadata = create_user_metadata(target_user_row, self.question_mapping, self.response_mapping)

        target_inputs = self.tokenizer(target_user_metadata, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            target_embedding = self.model(**target_inputs).last_hidden_state[:, 0, :].detach().cpu()

        similarities = [torch.nn.functional.cosine_similarity(target_embedding, user_embedding).item() for user_embedding in self.user_embeddings]

        similarities[target_user_idx] = -1

        top_k_indices = np.argsort(similarities)[-top_k:].tolist()

        return top_k_indices

    def find_similar_users_with_cosine_similarity(self, target_user_idx, top_k=5, balance=False):
        def calculate_cosine_similarity(target_row, comparison_row):
            mask = ~torch.isnan(target_row) & ~torch.isnan(comparison_row)
            if mask.sum() == 0:  # 비교할 유효한 값이 없을 때
                return float('nan')
            target_row_masked = target_row[mask]
            comparison_row_masked = comparison_row[mask]
            return torch.nn.functional.cosine_similarity(target_row_masked, comparison_row_masked, dim=0).item()

        df = pd.DataFrame(self.df, dtype=np.float32, copy=True)

        # print(target_user_idx)
        # print(True if target_user_idx in df.index else False)
        # print(df.index.tolist())
        target_user_row = torch.tensor(np.nan_to_num(df.loc[target_user_idx].values), dtype=torch.float32)
        # users_row_df = df.iloc[np.r_[0:target_user_idx, target_user_idx+1:len(df)]]
        users_row_df = df.drop(index=target_user_idx)
        similarities = [
            (index, calculate_cosine_similarity(
                target_user_row,
                torch.tensor(np.nan_to_num(user_row.values), dtype=torch.float32)
            )) for index, (_, user_row) in enumerate(users_row_df.iterrows())
        ]

        similarities = [item for item in similarities if not np.isnan(item[1])]
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 각 옵션별로 균형있게 뽑기, top_k 가 배수로 동작함 (예: top_k=2 -> 각 옵션별로 2명씩 뽑음), 순서는 A B C A B C 순으로 되도록 뽑음
        if balance:
            top_k_indices = []
            answer_count = {answer: 0 for answer in self.response_mapping[self.query_code] if answer != 99.0}
            total_needed = top_k * len(answer_count)

            answer_queue = list(answer_count.keys())  # 응답 카테고리 순서 리스트
            answer_index = 0

            for user_id, _ in similarities:
                if len(top_k_indices) >= total_needed:
                    break

                answer = self.df.loc[user_id][self.query_code]
                if answer == answer_queue[answer_index] and answer_count[answer] < top_k:
                    top_k_indices.append(user_id)
                    answer_count[answer] += 1
                    answer_index = (answer_index + 1) % len(answer_queue)

        else:
            top_k_indices = [item[0] for item in similarities[:top_k]]

        return top_k_indices

    def pad_or_trim_embedding(self, embedding, target_size):
        if embedding.shape[0] < target_size:
            padding_size = target_size - embedding.shape[0]
            padding = torch.zeros((padding_size, embedding.shape[1]))
            return torch.cat([embedding, padding], dim=0)
        else:
            return embedding[:target_size]

    def find_similar_response_users(self, target_user_idx, similar_responses, top_k=5):
        target_user_row = self.df.iloc[target_user_idx]
        target_user_responses = similar_responses[target_user_idx]

        target_inputs = self.tokenizer(target_user_responses, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            target_embedding = self.model(**target_inputs).last_hidden_state[:, 0, :].detach().cpu()

        # Calculate cosine similarities
        similarities = []
        for i, user_embedding in enumerate(self.user_response_embeddings):
            # Check if user_embedding is of correct shape [10, 1024]
            user_embedding = self.pad_or_trim_embedding(user_embedding, target_embedding.shape[0])
            if user_embedding.shape != target_embedding.shape:
                print(i, "번째 유저")
                raise ValueError(f"Shape of user_embedding {user_embedding.shape} does not match target_embedding {target_embedding.shape}")

            # Compute cosine similarity for each pair of embeddings and take the mean
            similarity = torch.nn.functional.cosine_similarity(target_embedding, user_embedding, dim=-1).mean().item()
            similarities.append(similarity)

        similarities[target_user_idx] = -1

        top_k_indices = np.argsort(similarities)[-top_k:].tolist()

        return top_k_indices

class BM25Retriever:
    def __init__(self, data_path="/data/matmang/ATP W117.sav", top_n=5):
        self.data_path = data_path
        self.top_n = top_n
        self.df, self.meta = self._load_data()
        self.question_mapping = self.meta.column_names_to_labels
        self.response_mapping = self.meta.variable_value_labels
        self.columns_to_drop = [
            'QKEY', 'INTERVIEW_START_W117', 'INTERVIEW_END_W117',
            'WEIGHT_W117', 'WEIGHT_W117_VOTE', 'LANG_W117',
            'FORM_W117', 'DEVICE_TYPE_W117', 'XTABLET_W117'
        ]
        self._preprocess_data()

    def _load_data(self):
        df, meta = pyreadstat.read_sav(self.data_path)
        return df, meta

    def _preprocess_data(self):
        self.df = self.df.drop(columns=self.columns_to_drop)

    def _get_question_and_answers_excluding_target(self, row, query_code):
        responses = []
        for question_code in row.index:
            if question_code == query_code:
                continue
            question = self.question_mapping.get(question_code, "Question code not found")
            # question_code와 일치하는 부분 제거
            if question_code in question:
                question = question.replace(question_code + ". ", "")
            answers = self.response_mapping.get(question_code, {})
            actual_answer = row[question_code]
            if pd.isna(actual_answer):
                continue
            actual_answer_text = answers.get(actual_answer, actual_answer) if isinstance(answers, dict) else actual_answer
            responses.append((question, actual_answer_text))
        return responses

    def select_relevants(self, responses, query, top_n=5):
        corpus = [f'Q : {r[0]} A : {r[1]}' for r in responses]

        tokenized_corpus = [doc.split() for doc in corpus]
        tokenized_query = query.split()

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
        responses_data = [(score, responses[i], i) for i, score in enumerate(scores)]

        sorted_results = sorted(responses_data, key=lambda x: x[0], reverse=True)
        top_responses = sorted_results[:top_n]

        ctxs = [(s[0], (s[1][0], s[1][1])) for s in top_responses]
        return ctxs

    def response_refinement(self, query, user, query_code):
        rsps = self._get_question_and_answers_excluding_target(user, query_code)
        results = self.select_relevants(responses=rsps, query=query, top_n=self.top_n)

        return results

if __name__ == '__main__':
    evaluator = BGERetriever()
    query = "Would you say that your vote for Congress in your district was more…"
    user = evaluator.df.iloc[0]
    responses = evaluator.response_refinement(query, user, query_code="VOTEFORAGNST_W117")
    print(responses)
