# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""

import csv
import json
import os

import datasets


_CITATION = """\
@InProceedings{supercon_dataset,
title={MDR SuperCon Datasheet},
author={Materials Database Group, National Institute for Materials Science},
year={2022}
doi={https://doi.org/10.48505/nims.3739}
}
"""

_DESCRIPTION = """\
Set of question answer pairs derived from the SuperCon dataset.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://doi.org/10.48505/nims.3739"

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL = "/lustre/isaac/proj/UTK0254/lp/superconductivity_dataset_cot.zip"
_URLS = {
    "train": _URL,
    "test": _URL,
    "val": _URL,
}

_LICENCE = """blah blah blah"""

class SuperconductivityDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="train", version=VERSION, description="This part of my dataset covers a first domain"),
        datasets.BuilderConfig(name="val", version=VERSION, description="This part of my dataset covers a first domain"),
        datasets.BuilderConfig(name="test", version=VERSION, description="This part of my dataset covers a first domain"),
    ]


    def _info(self):
        features = datasets.Features(
                {
                    "messages": [{"content": datasets.Value(dtype="string", id=None),
                                  "role": datasets.Value(dtype="string", id=None)}]
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        
        urls = _URLS[self.config.name]
        data_dir = os.path.join(dl_manager.download_and_extract(urls), "superconductivity_dataset_cot")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "datapath": os.path.join(data_dir, "train/"),
                    "jsonlpath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "datapath": os.path.join(data_dir, "val/"),
                    "jsonlpath": os.path.join(data_dir, "val.jsonl"),
                    "split": "val",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "datapath": os.path.join(data_dir, "test/"),
                    "jsonlpath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, datapath, jsonlpath, split):
        with open(jsonlpath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                paper_text = open(os.path.join(datapath, data["doi"].split("/")[1], 'out.txt'), 'r').read()

                # Defined here because it's helpful for playing with prompts. Will settle on a set prompt eventually. 
                sys_prompt = "You are a helpful assistant. You will answer questions about the following paper: {}".format(paper_text)
                guidelines = " Just answer the question separated by commas. Do not attempt to explain your answer. If you do not know the answer, write NA. If there are multiple materials studied, list the properties for them in a comma separated list, e.g. X, Y"
                chat_history = [{"role":"system","content": sys_prompt}]
                for message in data["messages"]:
                    chat_history.append({"role":"user", "content": message["question"] + guidelines})
                    chat_history.append({"role":"assistant", "content": message["answer"]})

                # Yields examples as (key, example) tuples
                yield key, {"messages": chat_history}
