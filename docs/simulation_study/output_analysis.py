"""
Script to generate tables shown in the paper.
"""

import optuna
import pandas as pd
import csv

CONFIG_ID = 1

# obtain list of datasets
datasets = []
with open('docs/simulation_study/datasets.csv', 'r', newline='') as csvfile:
    for i, row in enumerate(csv.reader(csvfile)):
        if int(row[3]) == CONFIG_ID:
            datasets.append(i)

val_lkh = []
test_lkh = []
datasets_ = []
method = []

for dataset_id in datasets:
    try:
        study_g = optuna.load_study(study_name=f"{dataset_id}_gaussian",
                                  storage="sqlite:///docs/simulation_study/sim_study.db")
    except KeyError as e:
        continue
    val_lkh.append(study_g.best_value)
    test_lkh.append(study_g.user_attrs["loss"])
    datasets_.append(dataset_id)
    method.append("gaussian")
    print(dataset_id)


for dataset_id in datasets:
    try:
        study_t = optuna.load_study(study_name=f"{dataset_id}_tukey",
                                    storage="sqlite:///docs/simulation_study/sim_study.db")
    except KeyError as e:
        continue
    val_lkh.append(study_t.best_value)
    test_lkh.append(study_t.user_attrs["loss"])
    datasets_.append(dataset_id)
    method.append("tukey")
    print(dataset_id)


import numpy as np
df = pd.DataFrame(dict(dataset_id=datasets_, method=method, val=val_lkh, test=test_lkh))

# Group by method and obtain summary statistics
result = df.groupby("method").agg(['mean', 'std', 'min', 'max'])
print(result[["val", "test"]].to_latex())

# Obtain likelihood ratios
pivot_df = df.pivot(index='dataset_id', columns='method')

# Calculate the ratios
ratios = {
    'val_ratio': - pivot_df['val']['tukey'] + pivot_df['val']['gaussian'],
    'test_ratio': - pivot_df['test']['tukey'] + pivot_df['test']['gaussian']
}

# Combine into a new DataFrame
result = pd.DataFrame(ratios).reset_index()
print(result.mean()[["val_ratio", "test_ratio"]])