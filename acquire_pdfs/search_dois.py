import pandas as pd
df = pd.read_csv('data/Superconductivity/data/220808_MDR_OAndM.txt', sep='\t', encoding='utf-8', encoding_errors='replace')

#title_names = df['title']
journal_names = df['journal']
physrevb_names_dirty = [n for n in journal_names if 'Phys.Rev.B' in n]
physrevb_names_dirty = list(set(physrevb_names_dirty))
physrevb_names = [n.split(',')[1].strip() for n in physrevb_names_dirty if ',' in n]
dois = []
for i, n in enumerate(physrevb_names):
    try:
        a = n.index("(")
        b = n.index(")")
        doi = (n[:a] + "." + n[b+1:])
        doi = "".join((n[:a] + "." + n[b+1:]).split()).lower()
        doi = "".join(doi.split("-")[0])
        dois.append(doi)
    except ValueError:
        continue

dois = list(set(dois))
    
valid_dois = []
with open('physrevb_dois.txt', 'r') as f:
    valid_dois = f.readlines()
valid_dois_stripped = [".".join(v.split(".")[-2:]).strip() for v in valid_dois]
valid_dois_map = {".".join(v.split(".")[-2:]).strip(): index for index, v in enumerate(valid_dois)}

output = []
bad_output = []
bad_output_index = []
for i, item in enumerate(dois):
    a = valid_dois_map.get(item)
    if a is None:
        bad_output.append(item)
        bad_output_index.append(i)
    else:
        output.append(a)

out = [valid_dois[i] for i in output if i is not None]
print(len(out))
with open('scihub_dois.txt', 'w') as f:
   for line in out:
        f.write(line)
