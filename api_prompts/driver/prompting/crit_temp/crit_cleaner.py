from pathlib import Path
import sys, json

path_root = Path(__file__).parents[2]
root = str(path_root)
sys.path.insert(1, root)


from API import jsonl_read

direc = "results/crit_temp_data_5"
results = jsonl_read(f"{direc}/results.jsonl")
new_results = results

for i, paper in enumerate(results):
    for j, chain in enumerate(paper["logs"]):
        for k, q in enumerate(chain['requests']):
            r = q['response']
            try:
                if not(r == "NA"):
                    g = r.split(",")
                    [float(i) for i in g]
            except ValueError:
                if r[-1] == "K":
                    r = r[:-1]
                
                if ")" in r:
                    index_end = r.find(")")

                    index_start = index_end-2

                    sub = r[index_start:index_end+1]

                    r = r.replace(sub, "")

                try:
                    if not(r == "NA"): 
                        g = r.split(",")
                        [float(i) for i in g]
                    
                    new_results[i]["logs"][j]["requests"][k]["response"] = r

                except ValueError:
                    raise ValueError(f"{r} : something is wrong : {paper}")


with open(f"{direc}/clean_results.jsonl", "w") as f:

    for i in new_results:
        f.write(json.dumps(i) + "\n")