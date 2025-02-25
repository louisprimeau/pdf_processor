# Author : Jackson Dendy
# Last Updated : 12/20/2024
# Description : Script create jsonl of critical temperature data all temp in K
import json

data = [{"doi": 97.094506, "data": [{"material": "LuRuB2", "temp": "9.8"},{"material": "YRuB2", "temp": '7.8'}]},
        {"doi": 72.024521, "data": [{"material": "Rb3C60", "temp": "30.7-30.9"}]},
        {"doi": 81.104522, "data": [{"material": "Ba8Si46", "temp": '8.1'}]},
        {"doi": 83.220512, "data": [{"material": "Ca-VII", "temp": '29'}]},
        {"doi": 56.9021, "data": [{"material": "YC2", "temp": '4.02'}]},
        {"doi": 51.12644, "data": [{"material": "H6Ni2B2C", "temp": '8.4'}]},
        {"doi": 87.224507, "data": [{"material": "SrPt2As2", "temp": '5'}]},
        {"doi": 68.104513, "data": [{"material": "MgB2-Poly Crystal", "temp": '38.5'}, {"material": "MgB2-Single Crystal", "temp": "36-38"}]},
        {"doi": 69.174503, "data": [{"material": "PbMo6S8", "temp": "15"}]},
        {"doi": 78.134504, "data": [{"material": "LuB12", "temp": '0.44'},{"material": 'ZrB12', "temp": '6'},{"material": "YB6", "temp": "6-7.5"}]},
        ]

new_data = []

for i in data:
    info = i['data']
    i.pop('data')
    i['messages'] = []

    mat_question = lambda mat: f"What is the criticl temperature of {mat} in the paper."

    for j in info:
        i['messages'].append({"question": mat_question(j['material']), 'answer': str(j['temp'])})

    i['doi'] =f"physrevb.{i['doi']}"
    new_data.append(i)

with open("crit_temp_questions.jsonl", "w") as f:

    for i in new_data:
        f.write(json.dumps(i) + "\n")