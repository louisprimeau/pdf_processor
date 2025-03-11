


import json
import os
import gzip
def list_json_files(directory):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return None
    files = os.listdir(directory)
    json_files = [file for file in files if file.endswith('.json.gz')]
    return json_files

source_directory = "/home/louis/data/crossref/April 2023 Public Data File from Crossref"
target_directory = "/home/louis/data/crossref/titles_dois"

for file_name in list_json_files(source_directory):
    out_list = []
    with gzip.open(os.path.join(source_directory, file_name), 'rt') as f:
        file_contents = json.load(f)

    for paper_dict in file_contents["items"]:
        if "title" in paper_dict.keys() and "DOI" in paper_dict.keys():
            out_list.append((paper_dict["title"][0], paper_dict["DOI"]))

    with open(os.path.join(target_directory, os.path.splitext(file_name)[0] + '.csv'), 'w') as f:
        for line in out_list:
            f.write("</sepchar_unused/>".join(line) + "\n")


