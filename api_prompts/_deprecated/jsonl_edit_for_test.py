from pathlib import Path
import sys, json

path_root = Path(__file__).parents[1]
root = str(path_root)
sys.path.insert(1, root)

from API import jsonl_read

data = jsonl_read("test.jsonl")

new_data = []

for i in data:

    point = i

    file = point["doi"]

    file = file[8:]

    point['doi'] = file

    new_data.append(point)

with open("new_test.jsonl", "w") as f:

    for i in new_data:
        f.write(json.dumps(i) + "\n")