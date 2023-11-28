# -*- coding: utf-8 -*-
from typing import Any

from torch.utils.data import Dataset


class RAGDataset(Dataset):
    prompt: str = None

    def __init__(self, data):
        self.data = data
        self.len = len(data)

    def preprocess(self, text: str) -> str:
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def __getitem__(self, index):
        return {
            "texts": self.fill_one_sample(self.data[index]),
            "iris": [
                self.data[index]["source"]["iri"],
                self.data[index]["target"]["iri"],
            ],
        }

    def __len__(self):
        return self.len

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        return self.prompt.replace("{source}", source).replace("{target}", target)

    def collate_fn(self, batchs):
        batchs_clear = {"texts": [], "iris": []}
        for batch in batchs:
            batchs_clear["texts"].append(batch["texts"])
            batchs_clear["iris"].append(batch["iris"])
        return batchs_clear


class LabelRAGDataset(RAGDataset):
    prompt = """Classify if two concepts refer to the same real world entity or not.
### First concept:
{source}
### Second concept:
{target}
### Answer:"""

    def __init__(self, data, llm_id):
        super().__init__(data=data)
        if "OpenAI" in llm_id:
            self.prompt = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
{source}
### Second concept:
{target}
### Answer:"""


class LabelParentRAGDataset(RAGDataset):
    pass


class LabelChildrenRAGDataset(RAGDataset):
    pass
