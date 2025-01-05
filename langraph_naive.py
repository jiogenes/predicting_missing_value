import itertools
import os
import re
import pickle
import pprint
from typing import TypedDict
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from sklearn.metrics import classification_report

from retrievers import BGERetriever

from discord import Webhook
import aiohttp
import asyncio
import inspect

async def send_discord_alert(message):
    file_name = inspect.getfile(inspect.currentframe()).split('/')[-1]
    async with aiohttp.ClientSession() as session:
        webhook = Webhook.from_url('https://discord.com/api/webhooks/', session=session)
        await webhook.send(message, username=file_name)

class GraphState(TypedDict):
    question: str
    user_response: list[str]
    useful_response: list[str]
    visited: list[str]
    user: int
    options: list[str]
    answer: str
    prompt: str
    useful_question: str
    few_shot_users: list[int]
    few_shot_responses: list[str]
    few_shot_actual_answers: list[str]
    few_shot_useful_responses: list[str]
    few_shot_demographics: list[str]
    top_k: int
    n_shot: int

def generate_llm_answer(messages, max_new_tokens):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
    )

    prompt = outputs[0][:input_ids.shape[-1]]
    response = outputs[0][input_ids.shape[-1]:]

    prompt = tokenizer.decode(prompt, skip_special_tokens=True)
    response = tokenizer.decode(response, skip_special_tokens=True)

    return prompt, response


# Few-shot 응답자 선정 및 응답자 응답 추출
def retrieve_responses(state: GraphState) -> GraphState:
    question = retriever.meta.column_names_to_labels[query_code].replace(f'{query_code}. ', '').strip()
    # using cosine similarity to find similar users
    state['few_shot_users'] = retriever.find_similar_users_with_cosine_similarity(state["user"], top_k=state['n_shot'])
    few_shot_demographics = []
    few_shot_responses = []
    few_shot_useful_responses = []
    few_shot_actual_answers = []
    cache_path = "cache/naive_qna/"
    options = [option for option in retriever.meta.value_labels[retriever.meta.variable_to_label[query_code]].values() if "Don't know/No Answer" not in option and "Refused" not in option and "Other" not in option]

    # naive question
    cache = os.path.join(cache_path, f'naive_qna_{query_code}.pkl')
    if os.path.exists(os.path.join(cache_path, f'naive_qna_{query_code}.pkl')):
        with open(os.path.join(cache_path, f'naive_qna_{query_code}.pkl'), 'rb') as f:
            bge_retrieved_qnas = pickle.load(f)
    else:
        print("fail")

    for user in state["few_shot_users"]:
        few_shot_responses.append(bge_retrieved_qnas[user][0][:state['top_k']])
        few_shot_actual_answers.append(retriever.response_mapping[query_code][retriever.df.loc[user][query_code]])
        few_shot_useful_responses.append([])

    return GraphState(
        question=question,
        options=options,
        user_response=bge_retrieved_qnas[state["user"]][0][:state['top_k']],
        user=state["user"],
        few_shot_users=state["few_shot_users"],
        few_shot_demographics=few_shot_demographics,
        few_shot_responses=few_shot_responses,
        few_shot_actual_answers=few_shot_actual_answers,
        visited=[],
        few_shot_useful_responses=few_shot_useful_responses)


def llm_answer(state: GraphState) -> GraphState:
    question = state["question"]
    user_response = '\n'.join(state["user_response"])
    options = '\n'.join(state["options"])

    messages = [
        {"role": "system", "content": f"You are tasked with predicting responses to targeted user survey questions through given user survey questions-responses. Read the provided user survey questions-responses and use it to select the most appropriate response from the given options to the target question. Ensure that your output includes only the selected response and does not include any additional comments, explanations, or questions. Choose the appropriate answer to the following target question from the following options. \n\nTarget question:\n{question}\n\nOptions:\n{options}"},
        {"role": "user", "content": f"Now, read the following target user survey responses and query, and select the most appropriate response from the given options based on the other responses.\nRefer to the answers provided by respondents similar to the user provided above.\nEnsure that your output includes only in Options:\nUser survey responses:\n{user_response}\n\nAnswer:"},
    ]

    # few-shot 추가하는 부분
    if state['few_shot_responses']:
        few_shot_examples = "Here are examples of respondents similar to this user:\n"

        for idx, (response, answer) in enumerate(zip(state["few_shot_responses"], state["few_shot_actual_answers"])):
            responses = '\n'.join(response)
            few_shot_examples += f"User {idx + 1}'s survey responses:\n{responses}\n\nAnswer:{answer}\n\n"

        messages.insert(1, {"role": "user", "content": few_shot_examples})

    prompt, response = generate_llm_answer(messages=messages, max_new_tokens=256)
    answer = response

    matches = []
    for option in state["options"]:
        general_match_string = re.search(r'\b' + re.escape(option) + r'\b', answer, re.IGNORECASE)
        if general_match_string:
            matches.append(True)
            exact_match_string = general_match_string.group(0)
            answer = exact_match_string
            break
        else:
            matches.append(False)

    if not any(matches):
        answer = 'None of the options match'

    return GraphState(answer=answer, prompt=prompt)

def main(args):
    global query_code, retriever, tokenizer, model, prompts

    query_code = args['query_code']

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_responses)
    workflow.add_node("llm_answer", llm_answer)
    workflow.add_edge("retrieve", "llm_answer")
    workflow.set_entry_point("retrieve")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    generated_answers = []
    prompts = []
    failed_user = []

    if query_code == 'SATIS_W116' or query_code == 'POL1JB_W116':
        index_1 = retriever.df[retriever.df[query_code] == 1.0].sample(n=50, random_state=1).index.tolist()
        index_2 = retriever.df[retriever.df[query_code] == 2.0].sample(n=50, random_state=1).index.tolist()
        user_ids = index_1 + index_2

    else:
        index_1 = retriever.df[retriever.df[query_code] == 1.0].sample(n=33, random_state=1).index.tolist()
        index_2 = retriever.df[retriever.df[query_code] == 2.0].sample(n=33, random_state=1).index.tolist()
        index_3 = retriever.df[retriever.df[query_code] == 3.0].sample(n=34, random_state=1).index.tolist()
        user_ids = index_1 + index_2 + index_3

    config = RunnableConfig(
        recursion_limit=18, configurable={"thread_id": "QUE-SEARCH-RAG"}
    )

    for user_id in tqdm(user_ids, desc="유저 응답 예측", ncols=80):
        inputs = GraphState(
            user=user_id,
            top_k=args['top_k'],
            n_shot=args['n_shot'],
        )
        try:
            output = app.invoke(inputs, config=config)
            generated_answers.append(output["answer"])
            prompts.append(output['prompt'])
        except GraphRecursionError as e:
            failed_user.append(user_id)
            generated_answers.append(output["answer"])
            prompts.append(output['prompt'])
            print(f"Recursion limit reached: {e}")

    real_answers = []
    for idx, user in enumerate(retriever.df.iloc[user_ids].itertuples(index=False)):
        if idx == 100:
            break
        user = pd.Series(user, index=retriever.df.columns)
        if pd.isna(user[query_code]):
            real_answers.append('Nan')
            continue
        real_answer = retriever.meta.variable_value_labels[query_code][user[query_code]]
        real_answers.append(real_answer.strip().lower())

    valid_indices = [idx for idx, answer in enumerate(real_answers) if answer not in ["don't know/no answer", "refused", "other", "nan"]]

    real_answers = [real_answers[idx] for idx in valid_indices]
    generated_answers = [generated_answers[idx].strip().lower() for idx in valid_indices]

    combined_output = []
    for prompt, generated_answer, real in zip(prompts, generated_answers, real_answers):
        combined_output.append(prompt)
        combined_output.append('generated: '+ generated_answer + ' / real: ' + real)

    with open(os.path.join('./', f'answers_{query_code}_{args["top_k"]}_{args["n_shot"]}.txt'), 'w') as f:
        f.write('\n\n'.join(combined_output))

    report = classification_report(real_answers, generated_answers, output_dict=True, zero_division=0)
    macro_avg_f1 = report['macro avg']['f1-score']
    print(f'F1 score : {macro_avg_f1}')
    return macro_avg_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_code", type=str, required=True)
    parser.add_argument("--top_k", type=int, nargs='+', required=True)
    parser.add_argument("--n_shot", type=int, nargs='+', required=True)
    parser.add_argument("--output_file", type=str, default='result')
    args = parser.parse_args()

    print(f'Parsed arguments: {args}')

    query_code = args.query_code
    param_grid = {
        'top_k': args.top_k,
        'n_shot': args.n_shot
    }

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    retriever = BGERetriever(top_n=10, data_path="./W116_Oct22/ATP W116.sav", query_code=query_code)
    prompts = []

    results = []
    for top_k, n_shot in itertools.product(*param_grid.values()):
        try:
            print(f"Running with parameters: top_k={top_k}, n_shot={n_shot}")
            params = {'query_code': query_code, 'top_k': top_k, 'n_shot': n_shot}
            macro_avg_f1 = main(params)
            results.append((top_k, n_shot, macro_avg_f1))
        except Exception as e:
            error_message = f"Error occurred with parameters: {params}\nException: {str(e)}"
            asyncio.run(send_discord_alert(message=error_message))
            raise

    # 결과를 데이터프레임으로 변환
    df_results = pd.DataFrame(results, columns=['top_k', 'n_shot', 'macro_avg_f1'])
    df_results.to_csv(args.output_file + f'_{query_code}_{args.top_k}_{args.n_shot}.csv', index=False)

    # 결과 출력
    print(df_results)

    # 그래프 그리기
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # 그래프 그리기
    palette = sns.color_palette("husl", len(df_results['n_shot'].unique()))
    line_styles = ['-', '--', '-.', ':']

    for n_shot, style in zip(df_results['n_shot'].unique(), line_styles):
        subset = df_results[df_results['n_shot'] == n_shot]
        sns.lineplot(data=subset, x='top_k', y='macro_avg_f1', label=f"n_shot={n_shot}", linestyle=style, marker='o', dashes=False)

    plt.title('Macro Avg F1-Score for Different Hyperparameter Combinations', fontsize=16)
    plt.xlabel('Top K', fontsize=14)
    plt.ylabel('Macro Avg F1-Score', fontsize=14)
    plt.legend(title='N Shot', fontsize=12, title_fontsize=14, loc='best')
    plt.grid(True)

    try:
        output_file = args.output_file + f'_{query_code}_{args.top_k}_{args.n_shot}.png'
        plt.savefig(output_file)
    except:
        plt.savefig(args.output_file + f'_{query_code}.png')

    asyncio.run(send_discord_alert(message='Done!'))