import os
from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
import pickle
from argparse import ArgumentParser
import torch

from sklearn.metrics import classification_report

from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from retrievers import BGERetriever
import useful_questions

categories = [
'F_METRO',
 'F_CREGION',
 'F_CDIVISION',
 'F_USR_SELFID',
 'F_AGECAT',
 'F_GENDER',
 'F_EDUCCAT',
 'F_EDUCCAT2',
 'F_HISP',
 'F_HISP_ORIGIN',
 'F_YEARSINUS_RECODE',
 'F_RACECMB',
 'F_RACETHNMOD',
 'F_CITIZEN',
 'F_BIRTHPLACE',
 'F_MARITAL',
 'F_RELIG',
 'F_BORN',
 'F_RELIGCAT1',
 'F_ATTEND',
 'F_PARTY_FINAL',
 'F_PARTYLN_FINAL',
 'F_PARTYSUM_FINAL',
 'F_PARTYSUMIDEO_FINAL',
 'F_VOTED2020',
 'F_VOTEGEN2020',
 'F_INC_SDT1',
 'F_IDEO',
 'F_INTFREQ',
 'F_VOLSUM',
 'F_INC_TIER2',
]

llm = ChatOllama(
    model="llama3:latest",
    temperature=0
)

generate_related_questions_template = '''Your task is to generate a list of questions based on the provided user input from a survey. Assume the user has completed as survey with various questions, but only one question is provided to you. Here's a description of the survey:

The ATP W116 survey, conducted by Pew Research Center, is a comprehensive pre-election questionnaire targeting a wide array of political and social issues. It was fielded from October 10 to October 16, 2022. The survey includes questions designed to gauge respondents' satisfaction with the current state of the country, approval ratings of President Joe Biden, opinions on various institutions, and perspectives on upcoming congressional elections.

Key sections of the survey include:
1. **Political Approval and Satisfaction**: Respondents are asked about their satisfaction with the country's direction and their approval or disapproval of President Biden's performance, including the strength of their opinions.
2. **Institutional Impact**: Questions explore perceptions of whether various institutions (e.g., churches, schools, technology companies, the military) have a positive or negative impact on the country.
3. **Election Engagement and Preferences**: Several questions assess how much thought respondents have given to the congressional elections, their voting plans, and preferences for congressional candidates. This section also probes the importance of specific issues (e.g., abortion, the economy, health care) in influencing voting decisions.
4. **Perceptions of Political Campaigns**: Respondents rate the effectiveness of Republican and Democratic campaigns in explaining their plans and visions.
5. **Importance of Issues**: A broad range of issues is covered to determine their importance in the upcoming congressional elections. These include economic conditions, healthcare, racial issues, and investigations into the actions of past and current administrations.
6. **Voting Logistics and Confidence**: Questions address how respondents plan to vote (in-person, absentee, or mail-in), their confidence in the vote-counting process, and perceptions of the ease and fairness of election rules.
7. **Social and Economic Opinions**: The survey delves into views on government assistance, military power, benefits of social policies, and other societal issues like the gender wage gap, government regulation, and national identity.
8. **International Relations**: Respondents are asked about their views on U.S. foreign policy, particularly regarding relations with China and Taiwan, and the significance of issues like China's military power and human rights policies.
9. **Historical Events**: There are questions about the January 6 Capitol riot, including perceptions of the attention it has received, the fairness of the investigation, and the responsibility of Donald Trump.
10. **Personal and Employment Situations**: The survey includes sections on respondents' current work status, pressures felt in their personal and professional lives, and their perceptions of economic issues affecting the nation and themselves personally.
Overall, the ATP W116 survey aims to capture a detailed snapshot of public opinion on a broad spectrum of topics leading up to the 2022 congressional elections.

The example of the provided questions is:
Thinking about the state of the country these days, would you say you feel...

Then you would generate additional questions such as:
How satisfied are you with the current direction of the country?
Do you approve or disapprove of President Biden’s performance?
How strongly do you feel about your approval or disapproval of President Biden?
Do you believe that the economy is improving, staying the same, or getting worse?
How much thought have you given to the upcoming congressional elections?
Are you planning to vote in-person, absentee, or by mail-in ballot?
How confident are you in the vote-counting process for the upcoming elections?
Do you think election rules are fair and make it easy to vote?
How would you rate the impact of technology companies on the country?
How important is the issue of healthcare in influencing your vote in the upcoming elections?
How effective do you think the Republican campaigns have been in explaining their plans and visions?
Do you believe that government assistance programs are beneficial to society?
How do you view the U.S. foreign policy towards China?
What are your thoughts on the significance of China’s human rights policies?
How would you rate the media’s coverage of the January 6 Capitol riot?
Do you think Donald Trump is responsible for the January 6 Capitol riot?
How do you feel about your current work status?
Do you feel pressured in your personal or professional life?
What are your perceptions of the economic issues affecting the nation?
How important is the issue of racial equality in influencing your vote in the upcoming elections?

Now, generate 20 useful questions for the following question.

Here's how you should proceed:
1. Analyze the provided question to understand its theme and context.
2. Generate additional relevant questions that would logically accompany the provided question in a survey.
3. Ensure the questions cover a wide range of aspects related to the theme of the provided question.
4. Do not add questions that are too similar to the provided question.
5. Do not add options or answer choices to the questions, only the questions themselves.
6. Do not say anything other than questions in your response.
7. You have to number the questions sequentially.

Provided question is:
{question}

Generate the additional survey questions:
'''

generate_answer_template = """
You are tasked with predicting the user’s response to a given previous user survey responses. 
Read the provided user survey responses and use it to select the most appropriate response from the given options. 
Ensure that your output includes only the selected response and does not include any additional comments, explanations, or questions.
Choose the appropriate answer to the last question from the options.

Here are examples of respondents similar to this user:
{few_shot_examples}

Now, read the following user survey responses and query, and select the most appropriate response from the given options based on the responses.
Refer to the answers provided by respondents similar to the user provided above.
Ensure that your output includes only in Options:

User survey responses:
{user_survey_responses}

Query:
{query}

Options:
{options}

Answer:
"""

def generate_related_questions(query):
    prompt = PromptTemplate.from_template(generate_related_questions_template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": query})

    assert result
    return result

def extract_questions_in_response(result):
    matches = []
    pattern = re.compile(r'\d+\.\s.*?(?=\d+\.\s|$)', re.DOTALL)
    for line in result.split('\n'):
        match = pattern.findall(line)
        if match:
            match = match[0].split(".", 1)[1].strip()
            matches.append(match)

    assert matches
    return matches

def extract_useful_qna(query, query_code):
    related_questions = getattr(useful_questions, query_code)
    extracted_questions = extract_questions_in_response(related_questions)
    result = retrieve_qna(extracted_questions, query_code, 100)
    return result

def extract_options_in_question(query_code):
    options = [option for option in retriever.meta.value_labels[retriever.meta.variable_to_label[query_code]].values() if "Don't know/No Answer" not in option and "Refused" not in option and "Other" not in option]

    return options

def get_question_and_answers(row, question_code):
    question = retriever.question_mapping.get(question_code, "Question code not found")
    answers = retriever.response_mapping.get(question_code, "No answers available for this question code")
    actual_answer = row[question_code] if question_code in row else "No answer provided"
    actual_answer_text = answers.get(actual_answer, actual_answer) if isinstance(answers, dict) else actual_answer

    answers = {k: v for k, v in answers.items() if v != 'Refused' or v != 'DK/Refused/No lean'}
    return question, answers, actual_answer_text

def create_user_metadata(row):
    sentence_parts = []
    for category in categories:
        question, _, answer = get_question_and_answers(row, category)
        if pd.isna(answer):
            continue
        sentence_parts.append(f"{question}: {answer}")
    return ", ".join(sentence_parts)

def extract_similar_users(top_k=5):
    results = []
    for index in tqdm(user_ids, total=100, ncols=80):
        retrieved_users = retriever.find_similar_users(index, top_k=top_k)
        results.append(retrieved_users)
    
    return results

def retrieve_qna(questions, query_code, top_k=10):
    results = []
    query_embeddings = []

    for question in questions:
        query_inputs = retriever.tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(retriever.device)
        with torch.no_grad():
            query_embedding = retriever.model(**query_inputs).last_hidden_state[:, 0, :].detach().cpu()
        query_embeddings.append(query_embedding)
    
    for user in tqdm(retriever.df.itertuples(index=False), total=len(retriever.df), ncols=80):
        result = []
        for question, query_embedding in zip(questions, query_embeddings):
            user = pd.Series(user, index=retriever.df.columns)
            retrieved_qna, _ = retriever.response_refinement(query=question, user=user, query_code=query_code, query_cache=query_embedding, top_n=top_k)
            for i in range(len(retrieved_qna)):
                if retrieved_qna[i] not in result:
                    result.append(retrieved_qna[i])
                    break  # Only add the first non-existing element
        results.append(result)

    return results

def generate_answer(query, similar_response, few_shot, options):
    user_responses = '\n'.join(similar_response)

    prompt = PromptTemplate.from_template(generate_answer_template)
    chain = prompt | llm | StrOutputParser()
    prompt_data = {
        "few_shot_examples": few_shot,
        "user_survey_responses": user_responses,
        "query": query,
        "options": options
    }
    generated_prompt = prompt.format(**prompt_data)
    result = chain.invoke({"few_shot_examples": few_shot, "user_survey_responses": user_responses, "query": query, "options": options})

    return result, generated_prompt

def generate_few_shot_examples(query, query_code, similar_users, similar_responses):
    few_shot_examples = []
    for index, i in enumerate(similar_users):
        s_user = retriever.df.iloc[i]
        s_user_rsps = similar_responses[i]
        s_user_rsps = f"**User {index + 1}'s example**\n" + 'User survey responses:\n' + '\n'.join(s_user_rsps)
        _, answers, actual_answer = get_question_and_answers(s_user, query_code)
        options = '\n'.join(extract_options_in_question(query_code))
        s_user_rsps += '\n\nQuery:\n' + query + '\n\nOptions:\n' + options + '\n\nAnswer:\n' + actual_answer
        few_shot_examples.append(s_user_rsps)

    examples = '\n\n'.join(few_shot_examples)

    return examples

def extract_answers(similar_users, similar_responses, query, query_code):

    options = extract_options_in_question(query_code)
    prompt_options = '\n'.join(options)
    extracted_answers = []
    raw_answers = []
    invalid_idx = []
    prompts = []
    for idx, user_id in tqdm(enumerate(user_ids), total=len(range(100)), ncols=80):
        similar_response = similar_responses[user_id]
        few_shots = generate_few_shot_examples(query, query_code, similar_users[idx], similar_responses)
        generated_answer, prompt = generate_answer(query, similar_response, few_shots, prompt_options)
        generated_answer = generated_answer.strip()

        matches = []
        for option in options:
            llama_match_string = re.search(r'\*\*' + re.escape(option) + r'\*\*', generated_answer, re.IGNORECASE)
            if llama_match_string:
                matches.append(True)
                exact_match_string = llama_match_string.group(0)
                exact_match_string = exact_match_string.replace('**', '')
                extracted_answers.append(exact_match_string)
                raw_answers.append(generated_answer)
                prompts.append(prompt)
                break
        
        if not any(matches):
            for option in options:
                general_match_string = re.search(r'\b' + re.escape(option) + r'\b', generated_answer, re.IGNORECASE)
                if general_match_string:
                    matches.append(True)
                    exact_match_string = general_match_string.group(0)
                    extracted_answers.append(exact_match_string)
                    raw_answers.append(generated_answer)
                    prompts.append(prompt)
                    break
                else:
                    matches.append(False)

        if not any(matches):
            print(f'LLM has generated invalid option... : {generated_answer}')
            invalid_idx.append(idx)

    return extracted_answers, invalid_idx, prompts, raw_answers
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--query_code', type=str, default='POL10_W116')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--cache_dir', type=str, default='cache')
    args = parser.parse_args()

    retriever = BGERetriever(top_n=100, data_path="./W116_Oct22/ATP W116.sav", query_code=args.query_code)
    
    # If you want to sample another user, you can change the user_ids
    if args.query_code == 'POL10_W116':
        # Sample 33 rows of each value 1.0 and 2.0, and 34 rows of value 3.0
        filtered_df = retriever.df[retriever.df[args.query_code].isin([1.0, 2.0, 3.0])]
        sampled_df_1_with_index = filtered_df[filtered_df[args.query_code] == 1.0].sample(n=33, random_state=2)
        sampled_df_2_with_index = filtered_df[filtered_df[args.query_code] == 2.0].sample(n=33, random_state=2)
        sampled_df_3_with_index = filtered_df[filtered_df[args.query_code] == 3.0].sample(n=34, random_state=2)
        balanced_sample_df_with_index = pd.concat([sampled_df_1_with_index, sampled_df_2_with_index, sampled_df_3_with_index])
    else:
        # Sample 50 rows of each value while keeping the original index
        filtered_df = retriever.df[retriever.df[args.query_code].isin([1.0, 2.0])]
        sampled_df_1_with_index = filtered_df[filtered_df[args.query_code] == 1.0].sample(n=50, random_state=2)
        sampled_df_2_with_index = filtered_df[filtered_df[args.query_code] == 2.0].sample(n=50, random_state=2)
        balanced_sample_df_with_index = pd.concat([sampled_df_1_with_index, sampled_df_2_with_index])

    # Sort the resulting DataFrame to maintain original index order
    balanced_sample_df_with_index = balanced_sample_df_with_index.sort_index()

    # Get the positional indices (row numbers) of the balanced DataFrame
    positional_indices = [retriever.df.index.get_loc(idx) for idx in balanced_sample_df_with_index.index]

    user_ids = positional_indices

    cache_path = Path(args.cache_dir).joinpath(f'useful_description')
    Path.mkdir(cache_path, exist_ok=True)

    print(f'Loading data... {args}')

    query_code = args.query_code
    query = retriever.meta.column_names_to_labels[query_code].replace(f'{query_code}. ', '').strip()

    if os.path.exists(os.path.join(cache_path, f'useful_qna_{query_code}_100.pkl')):
        print(f'Loading useful qna from cache...')
        with open(os.path.join(cache_path, f'useful_qna_{query_code}_100.pkl'), 'rb') as f:
            useful_qnas = pickle.load(f)
    else:
        print(f'Extracting useful qna...')
        useful_qnas = extract_useful_qna(query, query_code)
        with open(os.path.join(cache_path, f'useful_qna_{query_code}_100.pkl'), 'wb') as f:
            pickle.dump(useful_qnas, f)


