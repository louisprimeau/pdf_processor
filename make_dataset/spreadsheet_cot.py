import pandas as pd
import json
import re

df = pd.read_csv('220808_MDR_OAndM.txt', delimiter="\t",encoding='utf-8', encoding_errors='replace')

#g = {i:lambda x: ', '.join(list(x)) for i in df.columns[1:]}
df = df.fillna('NA').astype(str)
df = df.groupby(['journal'], as_index=False).agg(', '.join)

def clean_aggregate(string):
    ans = string.split(", ")
    ans = set(ans)
    if len(ans) > 1:
        while "NA" in ans:
            ans.remove("NA")
    return ", ".join(ans)

df_new = df.copy().applymap(clean_aggregate)
df_new['journal'] = df['journal']
df = df_new

fields = ['chemical formula', 'measured value of Oxygen content', 'temperature independent term in susceptibility', 'unit of CURIEC', 'Curie constant', 'unit of MOMENT', 'magnetic moment per formula', 'Curie temperature', 'Neel temperature', 'density (gcm-3)', 'unit of hardness ', 'hardness at 300 K ', "unit of Young's modulus ", "Young's modulus at 4.2 K ", "Young's modulus at 300 K ", 'unit of shear modulus ', 'shear modulus at 4.2 K ', 'shear modulus at 300 K ', 'unit of bulk modulus ', 'unit of bulk modulus at 4.2 K ', 'unit of bulk modulus at 300 K ', 'Poisson ratio at 4.2 K ', 'Poisson ratio at 300 K ', 'unit of sound velocity ', 'sound velocity at 4.2 K ', 'sound velocity at 300 K ', '*crystal structure, symmetry', 'common name of structure', 'space group', 'international table number', 'unit of lattice constant', 'lattice constant a', 'lattice constant b', 'lattice constant c', '*method of analysis for structure', 'unit of D(L)DT', 'temperature dependence of LATA', 'temperature dependence of LATB', 'temperature dependence of LATC', 'unit of D(L)DP', 'pressure dependence of LATA', 'pressure dependence of LATB', 'pressure dependence of LATC', 'unit of Tc', 'transition temperature (R = 0)', 'transition temperature (mid point)', 'transition temperature (R = 100%)', 'Tc from susceptibility measurement', 'lowest temperature for measurement (not superconducting)', 'transition width for resistive transition', 'Tc (of this sample) recommended', 'tc measurement method', 'unit of DTCDP', 'slope at P = 0 in Tc vs P plot ', 'maximum pressure applied', 'alpha in Tc = A * M^(-alpha), isotope effect', 'isotope element', 'exchange ratio of isotope(%)', 'DTC = Tc - Tc0 for isotope element', 'volume fraction of Meissner effect(%)', 'unit of Hc1', 'method of Hc1 derivation', 'Hc1 at 0 K for poly crystal', 'Hc1 at 0 K for single crystal for H //ab-plane', 'Hc1 at 0 K for single crystal for H //c-axis', 'Hc1 at given temperature for poly crystal', 'Hc1 at given temperature for single crystal H//ab-plane', 'Hc1 at given temperature for single crystal H//c-axis', 'measuring temperature', 'method of Hc2 derivation', 'Hc2 at 0 K for poly crystal', 'Hc2 at 0 K for single crystal for H //ab-plane', 'Hc2 at 0 K for single crystal for H //c-axis', 'Hc2 at given temperature for poly crystal', 'Hc2 at given temperature for single crystal H//ab-plane', 'Hc2 at given temperature for single crystal H//c-axis', 'measuring temperature.1', 'unit of dHc2/dT', 'method of dHc2/dT derivation', '-slope in Hc2 vs T at Tc for single crystal for H //ab-plane', '-slope in Hc2 vs T at Tc for single crystal for H //c-axis', 'difinition or method for Hirr', 'coherence length at 0 K for poly crystal', 'coherence length at 0 K for single crystal for H //ab-plane', 'method of PENET derivation', 'penetration depth at 0 K for poly crystal', 'penetration depth at 0 K for single crystal for H //ab-plane', 'unit of energy gap', 'energy gap at 0 K , delta(0)', 'normarized energy gap at 0 K , 2delta(0)/kTc', 'method of measuring energy gap', 'Jc at 4.2 K, H = 0 T', 'Jc at T = 77 K, H = 0 T', 'unit of SPJUMP', 'specific heat jump at Tc (delta-C)', 'unit of GAMMA', 'coefficient of electronic specific heat', 'Debye temperature', 'method for derivation of Debye temperature', 'unit of thermal conductivity', 'thermal conductivity at 300 K', 'thermal conductivity at 300 K for heat flow//c-axis', 'thermal conductivity at 300 K for heat flow//ab-plane', 'thermopower at 300 K', 'thermopower at 300 K for normal to ab-plane', 'thermopower at 300 K for parallel to ab-plane', 'unit of resistivity','resistivity at 4.2 K for poly crystal', 'resistivity at 4.2 K for single crystal for J//ab-plane', 'resistivity at 4.2 K for single crystal for J//c-axis', 'resistivity at 77 K for poly crystal', 'resistivity at 77 K for single crystal for J//ab-plane', 'resistivity at 77 K for single crystal for J//c-axis', 'resistivity at normal-T for poly crystal', 'resistivity at normal-T for single crystal for J//ab-plane', 'resistivity at normal-T for single crystal for J//c-axis', 'normal temperature', 'resistivity at RT for poly crystal', 'resistivity at RT for single crystal for J//ab-plane', 'resistivity at RT for single crystal for J//c-axis', 'unit of RH300', 'Hall coefficient at 300 K', 'Hall coefficient at 300 K for single, H//c-axis', 'Hall coefficient at 300 K for single, H//ab-plane', 'Hall coefficient for single, H//c-axis', 'magnetic field for Hall effect', 'unit of carrier density','carrier density at 300 K']


"""
new_fields = []
for field in fields:
    counts = df[field].value_counts('NA')
    if 'NA' not in counts.keys():
        new_fields.append(field)
        continue
    if counts['NA'] < 0.85:
        print(field, counts['NA'])
        new_fields.append(field)

fields = new_fields
"""

#fields = ['chemical formula', 'space group', 'unit of lattice constant', 'lattice constant a', 'lattice constant b', 'lattice constant c', 'unit of Tc', 'transition temperature (R = 0)', 'transition temperature (mid point)', 'transition temperature (R = 100%)', 'Tc from susceptibility measurement']

doi_to_spreadsheet = pd.read_csv('doi_map.txt', sep='^([^,]+),', engine='python')
journal_names_saved = list(doi_to_spreadsheet['journal_name'])
journal_dois_saved = list(doi_to_spreadsheet['doi'])

def use_map(column_a, column_b, a):
    return column_b[column_a.index(a)]

questions = ["What is the {} for the material(s) described in the paper?".format(field) for field in fields]

relevant_rows = []
for i, journal_name in enumerate(df['journal']):
    if journal_name in journal_names_saved:
        relevant_rows.append(i)

import random
random.seed(42)
NA_chance = 0.01
random.shuffle(relevant_rows)

train_split = relevant_rows[:int(len(relevant_rows) * 0.8)]
val_split = relevant_rows[int(len(relevant_rows) * 0.8):int(len(relevant_rows) * 0.9)]
test_split = relevant_rows[int(len(relevant_rows) * 0.9):]

with open('train.jsonl', 'w') as f:
    for i, row in enumerate(train_split):
        print("row {}, {}".format(i, use_map(journal_names_saved, journal_dois_saved, df.iloc[row]['journal'])))
        out_dict = {'doi': use_map(journal_names_saved, journal_dois_saved, df.iloc[row]['journal']),
                    'journal_name': df.iloc[row]['journal'],
                    'messages': [],
                    }
        for field, question in zip(fields, questions):
            answer = df.iloc[row][field]
            if answer == 'NA' and random.random() > NA_chance:
                continue
            if not pd.isna(answer):
                out_dict['messages'].append({
                            'question': question,
                            'answer': answer,
                                 })
        f.write(json.dumps(out_dict) + "\n")

with open('val.jsonl', 'w') as f:
    for row in val_split:
        out_dict = {'doi': use_map(journal_names_saved, journal_dois_saved, df.iloc[row]['journal']),
                    'journal_name': df.iloc[row]['journal'],
                    'messages': [],
                    }
        for field, question in zip(fields, questions):
            answer = df.iloc[row][field]
            if answer == 'NA' and random.random() > NA_chance:
                continue
            if not pd.isna(answer):
                out_dict['messages'].append({
                            'question': question,
                            'answer': answer,
                                 })
        f.write(json.dumps(out_dict) + "\n")

with open('test.jsonl', 'w') as f:
    for row in test_split:
        out_dict = {'doi': use_map(journal_names_saved, journal_dois_saved, df.iloc[row]['journal']),
                    'journal_name': df.iloc[row]['journal'],
                    'messages': [],
                    }
        for field, question in zip(fields, questions):
            answer = df.iloc[row][field]
            if answer == 'NA' and random.random() > NA_chance:
                continue
            if not pd.isna(answer):
                out_dict['messages'].append({
                            'question': question,
                            'answer': answer,
                                 })
        f.write(json.dumps(out_dict) + "\n")
