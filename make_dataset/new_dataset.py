import pandas as pd
import json
import re

df = pd.read_csv('220808_MDR_OAndM.txt', delimiter="\t",encoding='utf-8', encoding_errors='replace')

#g = {i:lambda x: ', '.join(list(x)) for i in df.columns[1:]}
df = df.fillna('NA').astype(str)
df = df.groupby(['journal'], as_index=False).agg(', '.join)

fields = ['chemical formula', 'Curie temperature', 'Neel temperature', 'space group', 'unit of lattice constant', 'lattice constant a', 'lattice constant b', 'lattice constant c', 'unit of Tc', 'transition temperature (R = 0)', 'transition temperature (mid point)', 'transition temperature (R = 100%)', 'Tc from susceptibility measurement', ]

doi_to_spreadsheet = pd.read_csv('doi_map.txt', sep='^([^,]+),', engine='python')
journal_names_saved = list(doi_to_spreadsheet['journal_name'])
journal_dois_saved = list(doi_to_spreadsheet['doi'])

def use_map(column_a, column_b, a):
    return column_b[column_a.index(a)]

questions = fields

relevant_rows = []
for i, journal_name in enumerate(df['journal']):
    print(journal_name)
    if journal_name in journal_names_saved:
        relevant_rows.append(i)

import random
random.seed(42)

random.shuffle(relevant_rows)

train_split = relevant_rows[:int(len(relevant_rows) * 0.8)]
val_split = relevant_rows[int(len(relevant_rows) * 0.8):int(len(relevant_rows) * 0.9)]
test_split = relevant_rows[int(len(relevant_rows) * 0.9):]

with open('train_new.jsonl', 'w') as f:
    for row in train_split:
        out_dict = {'doi': use_map(journal_names_saved, journal_dois_saved, df.iloc[row]['journal']),
                    'journal_name': df.iloc[row]['journal'],
                    'data': []}

        chem_formulas = df.iloc[row]["chemical formula"].split(", ")
        out_dict['data'] = [{"chemical formula": ch} for ch in chem_formulas]
        for field in fields[1:]:
            answers = df.iloc[row][field].split(", ")
            for i, answer in enumerate(answers):
                if answer != "NA":
                    out_dict['data'][i][field] = answer

        f.write(json.dumps(out_dict) + "\n")

with open('val_new.jsonl', 'w') as f:
    for row in val_split:
        out_dict = {'doi': use_map(journal_names_saved, journal_dois_saved, df.iloc[row]['journal']),
                    'journal_name': df.iloc[row]['journal'],
                    'data': []}

        chem_formulas = df.iloc[row]["chemical formula"].split(", ")
        out_dict['data'] = [{"chemical formula": ch} for ch in chem_formulas]
        for field in fields[1:]:
            answers = df.iloc[row][field].split(", ")
            for i, answer in enumerate(answers):
                if answer != "NA":
                    out_dict['data'][i][field] = answer
        f.write(json.dumps(out_dict) + "\n")

with open('test_new.jsonl', 'w') as f:
    for row in test_split:
        out_dict = {'doi': use_map(journal_names_saved, journal_dois_saved, df.iloc[row]['journal']),
                    'journal_name': df.iloc[row]['journal'],
                    'data': []}

        chem_formulas = df.iloc[row]["chemical formula"].split(", ")
        out_dict['data'] = [{"chemical formula": ch} for ch in chem_formulas]
        for field in fields[1:]:
            answers = df.iloc[row][field].split(", ")
            for i, answer in enumerate(answers):
                if answer != "NA":
                    out_dict['data'][i][field] = answer
        f.write(json.dumps(out_dict) + "\n")
