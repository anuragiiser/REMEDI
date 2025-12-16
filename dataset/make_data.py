import pandas as pd
import json
import numpy as np
import os

os.listdir()

mimic3_path = '/home/anurag/nkg/datasets/mimic_3/mimic3'

os.listdir(mimic3_path)

df = pd.read_csv(os.path.join(mimic3_path, 'symptoms_val.txt'))
# df = pd.read_csv('/home/anurag/nkg/datasets/mimic_4/mimic_iv/mimic-iv-preprocessed-icd-symptoms.csv')

df.head()

patients_ids = df['id'].tolist()
len(patients_ids)

# save patients ids to a json file
with open('patients_val_ids.json', 'w') as f:
    json.dump(patients_ids, f)

def make_3_digit_codes(string):
    codes = string.split(',')
    icd9_codes = []
    for code in codes:
        # if code starts with E
        if code.startswith('E'):
            code = code[:4]
        else:
            code = code[:3]
        icd9_codes.append(code)
    return icd9_codes

df['final_codes'] = df['short_codes'].apply(make_3_digit_codes)

final_codes = df['final_codes'].to_list()

# check frequency of each code
from collections import Counter
code_counter = Counter()
for codes in final_codes:
    code_counter.update(codes)

code_counter.most_common(10)

most_common_codes = [code for code, freq in code_counter.most_common(10)]
print(most_common_codes)

def filter_codes(icd9_codes, most_common_codes):
    filtered_codes = []
    for code in icd9_codes:
        if code in most_common_codes:
            filtered_codes.append(code)
    return filtered_codes

df['icd9_codes'] = df['final_codes'].apply(lambda x: filter_codes(x, most_common_codes))

# for the row in which icd9_codes is empty, drop that row
df = df[df['icd9_codes'].map(len) > 0]

df.head()

# save the dataframe to a csv file in mu_bench folder
df.to_csv('mu_bench/mimic.csv', index=False)

mimic_path = '/home/anurag/nkg/datasets/mimic_4/mimic_iv'

os.listdir(mimic_path)

file_path = os.path.join(mimic_path, 'mimic-iv_val.txt')
df = pd.read_csv(file_path)

df.head()

print(df['SHORT_CODES'][3])

count = 0
for i in range(len(df)):
    if '585' in df['SHORT_CODES'][i]:
        count += 1

print(count)

df.shape

# make a new df with only rows = 'hadm_id', 'Symptoms', 'TEXT', 'SHORT_CODES'
# and rename then as 'id', 'Symptoms', 'text', 'short_codes'

df_new = df[['hadm_id', 'Symptoms', 'TEXT', 'SHORT_CODES']].rename(
    columns={
        'hadm_id': 'id',
        'TEXT': 'text',
        'SHORT_CODES': 'short_codes'
    }
)

df_new.shape

# save the new df to a csv file
output_path = os.path.join(mimic_path, 'MIMIC4_VAL.csv')
df_new.to_csv(output_path, index=False)

