import pandas as pd
import csv
import json
import os
import gzip
import re

def list_csv_files(directory):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return None
    files = os.listdir(directory)
    csv_files = [file for file in files if file.endswith('.json.csv')]
    return csv_files

def process(string):

    string = string.replace('<sub>', '')
    string = string.replace('<\/sub>', '')
    regex = re.compile('[^a-zA-Z]')
    string = regex.sub('', string)
    return string.lower()

source_directory = "/home/louis/data/raw_data/Superconductivity/data"
source_title = "220808_MDR_OAndM.txt"
df = pd.read_csv(os.path.join(source_directory, source_title), delimiter='\t', encoding='utf-8', encoding_errors='replace')
title_names = list(df['title of reference'])


hash_table = {}
csv_directory = "/home/louis/data/crossref/titles_dois"
for csv_file in list_csv_files(csv_directory):
    df = pd.read_csv(os.path.join(csv_directory, csv_file), sep='</sepchar_unused/>', header=None, names=['paper_title', 'doi'], engine='python')
    for paper_title, doi in zip(df['paper_title'], df['doi']):
        hash_table[process(str(paper_title))] = doi

hits = []
non_hits = []
for i, title in enumerate(title_names[1:]):
    if process(str(title)) in hash_table.keys():
        hits.append((title, hash_table[process(str(title))]))
    else:
        non_hits.append(title)

unique_entries = {tup[1]: tup for tup in hits if tup[1] is not None and 'physrevb' not in tup[1]}.values()
        
with open('doi_title_map_2.txt', 'w') as f:
    for hit in unique_entries:
        f.write("{},{}\n".format(hit[0],hit[1])
        
#dois = set(entry[1] for entry in hits if entry[1] is not None and 'physrevb' not in entry[1])
"""
with open('all_dois.txt', 'w') as f:
    for doi in dois:
        f.write(doi + '\n')
"""
