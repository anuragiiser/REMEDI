import pandas as pd
import json
import numpy as np
import os
from tqdm import tqdm

df = pd.read_csv('/home/anurag/nkg/datasets/mu_bench/mimic.csv')

df.head()

df.iloc[0]['Symptoms']

print(df.iloc[0]['text'])

df.columns

print(df.iloc[0]['long_texts'])

print(df.iloc[0]['discharge_summary'])

def choose_random_rows_indexes(df, n, seed=42):
    np.random.seed(seed)
    return np.random.choice(df.index, size=n, replace=False)

seed_values = [('fold_1', 42)]
n_values = [64, 128, 256]

forget_set = {}

for n in n_values:
    forget_set[n] = {}
    for fold, seed in seed_values:
        indexes = choose_random_rows_indexes(df, int(n/2), seed=seed)
        forget_set[n][fold] = indexes.tolist()

forget_set.keys()

for n in n_values:
    set = forget_set[n]
    df_temp = df.drop(index=set['fold_1'])
    for fold, seed in seed_values:
        indexes = choose_random_rows_indexes(df_temp, int(n/2), seed=seed)
        #append indexes to forget set
        forget_set[n][fold] = set['fold_1'] + indexes.tolist()

len(forget_set[64]['fold_1'])

# save in json with proper formatting
with open('concurrent_forget_set.json', 'w') as f:
    json.dump(forget_set, f, indent=4)

"""overlap forget set"""

seed_values = [('fold_1', 41)]
n_values = [64, 128, 256, 512]

exact_overlap_forget_set = {}

for n in n_values:
    exact_overlap_forget_set[n] = {}
    for fold, seed in seed_values:
        indexes = choose_random_rows_indexes(df, n, seed=seed)
        exact_overlap_forget_set[n][fold] = indexes.tolist()

exact_overlap_forget_set

# save in json with proper formatting
with open('exact_overlap.json', 'w') as f:
    json.dump(exact_overlap_forget_set, f, indent=4)

"""# percentage instances"""

percentage_instances = [3,4,5]
n_values = [int(len(df)*p/100) for p in percentage_instances]
seed_values = [('fold_1', 42)]
print(n_values)

percentage_forget_set = {}

for n in n_values:
    percentage_forget_set[n] = {}
    for fold, seed in seed_values:
        indexes = choose_random_rows_indexes(df, n, seed=seed)
        percentage_forget_set[n][fold] = indexes.tolist()

percentage_forget_set

# save in json with proper formatting
with open('percentage_overlap.json', 'w') as f:
    json.dump(percentage_forget_set, f, indent=4)

bm25_overlap = {}
seed_values = [('fold_1', 40)]
n_values = [64, 128, 256, 512]

for n in n_values:
    bm25_overlap[n] = {}
    for fold, seed in seed_values:
        indexes = choose_random_rows_indexes(df, n, seed=seed)
        bm25_overlap[n][fold] = indexes.tolist()

# check best bm25 overlap

case = 21

df_indices = df.index.tolist()

from rank_bm25 import BM25Okapi

# check bm25 overlap between case and df indices
def bm25_overlap_indices(case, indices):
    case_text = df['text'][case]
    overlaps = []
    for idx in tqdm(indices):
        text = df['text'][idx]
        bm25 = BM25Okapi([case_text.split()])
        score = bm25.get_scores(text.split())[0]
        overlaps.append((idx, score))
    overlaps.sort(key=lambda x: x[1], reverse=True)
    return overlaps

df['text'][21]

overlaps = bm25_overlap_indices(case, df_indices)
overlaps[:10]

print(df['text'][32])

print(df['text'][overlaps[0][0]])

print(df['text'][6436])

df['icd9_codes'][6436]

