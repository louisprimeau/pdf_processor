import json

model_output_file = "Meta-Llama-3.1-405B-Instruct_test_4b_validation.jsonl"
#correct_answer_file = "/lustre/isaac/proj/UTK0254/lp/superconductivity_dataset_cot/val.jsonl"


def read_jsonl(file_path):
    data = [] 
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

responses = read_jsonl(model_output_file)

num_correct = 0
for response in responses:

    if response['response'] == response['answer']:
        num_correct += 1