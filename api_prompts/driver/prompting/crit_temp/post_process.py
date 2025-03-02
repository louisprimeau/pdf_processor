from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
root = str(path_root)
sys.path.insert(1, root)

import API 

metrics = {
        'confidence': [0, 100],  
        'LLM': [0, 100], 
        'E2E': [0, 1]
        }

API.batch_paper_pp("/home/jdendy/pdf_processor/api_prompts/prompting/crit_temp/results/crit_temp_data_5", metrics)