
text = """ Studies ofi the electrical transport properties--resistivity. Hali effect, and Scebeck
coefficient--of the hexagonal tungsten bronze Rb,WO, for 0.16 x <0.33 are reported for the temperature range from 1.5 to 300 K. In the normal state, these properties show temperaturedepenient anomalies which suggest a phase transition with temperature. The temperature at which this transition oecurs is shown to depend on the Rb concentration, exhibiting a sharp maximum near x -0.25. As the Rb concentration is decreased below x =0.33 the supercon. ducting transition temperature first increases above the value of 1.94 K $\textup{Kat}\,x=0.33$ and then decreases to less than 1 K at x =0.24. As the Rb concentration is lowered further a second increase in T is observed. This behavior appears to be associated with a phase transition at a $T_{c}$ 
composition near $x\,=0.25.$ The nature of neither the transition with temperature nor the transition with concentration could be precisely determined. The superconducting state shows a iarge temperature-independent $60^{\circ}$ anisotropy in $H_{c2}$ in the plane perpendicular to the hexagonal axis and positive curvature in the temperature dependence of $H_{c2}$ near $T_{c}$ 
 filling of the hexagonal channels, and most previous studies of the hexagonal phase of Rb.WO: have been at or near this upper limit
 The majority of the studies of the tungsten bronzes have centered on the cubic phase. Of the elements for which this phase is possible, Na.WOs has been studied in the most detail.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import datasets
import transformers
import torch
import json
import gc
from enum import Enum
from pydantic import BaseModel, constr, PositiveFloat, conint, conlist
from typing import Optional, Sequence, Tuple
import outlines
import datetime

class NumberWithUnit(BaseModel):
    number: float
    unit: constr(max_length=2)

class Material(BaseModel):
    chemical_formula: constr(max_length=30)
    #chemical_formula: ChemicalFormula
    critical_temperature: NumberWithUnit
    upper_critical_field: Optional[NumberWithUnit]
    neel_temperature: Optional[NumberWithUnit]
    lattice_constant_a: Optional[NumberWithUnit]
    lattice_constant_b: Optional[NumberWithUnit]
    lattice_constant_c: Optional[NumberWithUnit]
    
class MaterialList(BaseModel):
    materials: conlist(item_type=Material,min_length=1, max_length=5)
    
model_id = "google/gemma-2-27b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
outline_model = outlines.models.Transformers(model, tokenizer)
generator = outlines.generate.json(outline_model, MaterialList)

prompt = "You are a world class large language model tasked with extracting material information from passages of scientific papers. Please describe the materials described on the following passage: {text} and extract the chemical formula, superconducting critical temperature, upper critical field, the Neel temperature, the a lattice constant, the b lattice constant, and the c lattice constant for each material in order. There may be multiple materials studied, including similar materials with different stoichiometry. If you can't find the answer in the provided text, skip it."




out_file_name = f'{model_id.split("/")[1]}_{test_with}_{datetime.datetime.now().strftime("%Y-%m-%d%H:%M:%S")}.jsonl'

#with open(out_file_name, 'a') as f:
for i, message in enumerate(iterable_dataset):
    out = generator(prompt.format(text=message['messages'][0]['content'][:8000]))
    print(out.json())
    breakpoint()
    with open(out_file_name, "a") as f:
        f.write(out.json() + "\n")
        
    torch.cuda.empty_cache()
    gc.collect()
