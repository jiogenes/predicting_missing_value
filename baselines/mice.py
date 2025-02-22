import argparse
import pyreadstat
import numpy as np
import pandas as pd
import json
import os
from itertools import product
from copy import deepcopy
from tqdm import tqdm

import miceforest as mf
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")


exclude_columns = [
    'QKEY', 'INTERVIEW_START_W116', 'INTERVIEW_END_W116',
    'WEIGHT_W116', 'XW91NONRESP_W116', 'LANG_W116',
    'FORM_W116', 'DEVICE_TYPE_W116', 'XW78NONRESP_W116'
]

label_names = ['SATIS_W116', 'POL10_W116', 'POL1JB_W116', 'DRLEAD_W116', 'PERSFNCB_W116', 'ECON1B_W116']

def get_data(label_name, total):
    df, meta = pyreadstat.read_sav("/data/jyji/ATP W116.sav")

    n_users = int(len(df) * 0.5) # 0.5
    missing_df = df.sample(n=n_users, random_state=42)
    missing_user_ids = missing_df.index
    if label_name == 'SATIS_W116' or label_name == 'POL1JB_W116':
        index_1 = missing_df[missing_df[label_name] == 1.0].sample(n=50, random_state=42).index.tolist()
        index_2 = missing_df[missing_df[label_name] == 2.0].sample(n=50, random_state=42).index.tolist()
        user_ids = index_1 + index_2
    else:
        index_1 = missing_df[missing_df[label_name] == 1.0].sample(n=33, random_state=42).index.tolist()
        index_2 = missing_df[missing_df[label_name] == 2.0].sample(n=33, random_state=42).index.tolist()
        index_3 = missing_df[missing_df[label_name] == 3.0].sample(n=34, random_state=42).index.tolist()
        user_ids = index_1 + index_2 + index_3

    missing_user_ids = [i for i in missing_user_ids if i not in user_ids]

    df = df.drop(exclude_columns, axis=1)
    df = df.drop(missing_user_ids, axis=0)
    return df, user_ids

def kfold_train(X, index, label_name, datasets, max_iter):
    coloned = deepcopy(X)
    coloned.loc[index, label_name] = np.nan

    kds = mf.ImputationKernel(
        coloned,
        variable_schema=[label_name],
        datasets=datasets,
        save_all_iterations=False,
        random_state=42,
    )
    kds.mice(max_iter)
    imputed = kds.complete_data(variables=[label_name])

    y = X.loc[index, label_name].tolist()
    pred = imputed.loc[index, label_name].tolist()

    report = classification_report(y, pred, output_dict=True, zero_division=0.0)
    print(f"[{label_name}]\t f1-score: {report['macro avg']['f1-score']}")
    return report['macro avg']['f1-score']

def run_experiments(test_name, totals, datasets_list, max_iters):
    dest_dir = f"results/{test_name}"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    all_results = {}

    for total, datasets, max_iter in product(totals, datasets_list, max_iters):
        acc_dict = {}
        experiment_name = f"{test_name}_total{total}_datasets{datasets}_maxiter{max_iter}"
        print(f"Running experiment: {experiment_name}")

        for label_name in label_names:
            X, index = get_data(label_name, total)
            f1_score = kfold_train(X, index, label_name, datasets, max_iter)
            acc_dict[label_name] = f1_score

        with open(f"{dest_dir}/{experiment_name}.json", "w") as f:
            json.dump(acc_dict, f, indent=4)

        all_results[experiment_name] = acc_dict

    with open(f"{dest_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--totals", type=int, nargs='+', default=[100], help="List of total numbers of samples")
    parser.add_argument("--datasets_list", type=int, nargs='+', default=[2], help="List of dataset counts for MICE")
    parser.add_argument("--max_iters", type=int, nargs='+', default=[2], help="List of max iterations for MICE")
    args = parser.parse_args()

    run_experiments("MICE-50", args.totals, args.datasets_list, args.max_iters)