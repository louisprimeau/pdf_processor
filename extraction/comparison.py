import json
import chemparse
import os

with open('/lustre/isaac/proj/UTK0254/lp/pdf_processor/extraction/output/run_llama3_1_70B_28000chars_qs.json') as f:
    model_output = json.load(f)
with open('/lustre/isaac/proj/UTK0254/lp/pdf_processor/extraction/output/database.json') as f:
    database = json.load(f)

model_output = {key:[model_output[key][0][:2], model_output[key][1][0]] for key in model_output.keys() if key != 'questions'}

def format_material_string(string, splitchar):
    string = string.replace('MATERIAL:', '')
    string_list = "".join(string.split()).split(splitchar)
    return string_list
    
def format_temperature_string(temp_list):
    temp_list = "".join(temp_list.replace('K', '').split()).split(',')
    temp_list = [t for t in temp_list if len(t) > 0]
    temp_list = [temp_string[:-1] if temp_string[-1] == '.' else temp_string for temp_string in temp_list]
    temp_list_float = []
    for temp in temp_list:
        try:
            temp_list_float.append(float(temp))
        except ValueError:
            print(temp)
            temp_list_float.append(-100.0)
    return temp_list_float

def standard_chem_formula(dictionary):
    return ''.join(char for char in ''.join(key for key in sorted(dictionary.keys())) if char.isalpha() or char.isnumeric())

processed_model_output = {}
for key in model_output.keys():
    if key == 'questions': continue
    materials = format_material_string(model_output[key][0][0], '&')
    temperatures = format_temperature_string(model_output[key][0][1])
    if len(materials) - len(temperatures) != 0: continue
    materials = list(set(materials))
    indices = [materials.index(s) for s in materials]
    if len(materials) > len(temperatures):
        temperatures = temperatures + [-100.0] * (len(materials) - len(temperatures))
    temperatures = [temperatures[i] for i in indices]
    processed_model_output[key] = (materials, temperatures)
breakpoint()
processed_database_output = {}
for key in database.keys():
    materials = format_material_string(database[key][0], ',')
    temperatures = format_temperature_string(database[key][1].replace('CRITICAL TEMPERATURE:', ''))
    if len(materials) > len(temperatures):
        temperatures = temperatures + [-100.0] * (len(materials) - len(temperatures))
    processed_database_output[key] = (materials, temperatures)

shared_keys = list(sorted(processed_model_output.keys() & processed_database_output.keys()))
processed_model_output = {key:processed_model_output[key] for key in shared_keys}
processed_database_output = {key:processed_database_output[key] for key in shared_keys}


paper_source_directory = '/lustre/isaac/proj/UTK0254/lp/superconductivity_dataset/'
file_name = 'out.txt'

correct_num_materials = 0
for key in processed_database_output.keys():
    dbase_output = processed_database_output[key][1][0]
    model_db_output = processed_model_output[key][1][0]

    paper_textfile = os.path.join(paper_source_directory, key, file_name)
    with open(paper_textfile) as f:
        paper_text = f.read()
    if '"' in model_output[key][1]: model_output[key][1] = model_output[key][1].split('"')[1]
    
    sentence_real = model_output[key][1] in paper_text
    
    if abs(dbase_output - model_db_output) < 2.0:
        correct_num_materials += 1
        print("{: >20} {: >30} {: >10} {: >20} {} {}".format(key, processed_database_output[key][0][0], dbase_output, model_db_output, sentence_real, model_output[key][1], sep='\t'))
    else:
        print("{: >20} {: >30} {: >10} {: >20} {} {} {}".format(key, processed_database_output[key][0][0], dbase_output, model_output[key][0][1], "*", sentence_real, model_output[key][1]))
        
print(correct_num_materials / len(processed_database_output))
