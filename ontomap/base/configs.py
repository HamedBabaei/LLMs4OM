# -*- coding: utf-8 -*-
"""
    BaseConfig: Data Configuration of models
"""
import argparse
import os
from typing import Dict, Optional


class BaseConfig:
    def __init__(self, root_dataset_dir: str, approach: Optional[str] = "none"):
        self.root_dataset_dir = root_dataset_dir
        self.parser = argparse.ArgumentParser()
        self.approach = approach

    def mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def add_path(self):
        pass

    def __str__(self):
        return "config."

    def flan_t5(self, device: str) -> Dict:
        if self.approach == "baseline":
            config = {"max_token_length": 500, "num_beams": 10, "device": device}
        else:
            config = {}
        return config

    def llama(self, device: str) -> Dict:
        if self.approach == "baseline":
            config = {"max_token_length": 500, "num_beams": 10, "device": device}
        else:
            config = {}
        return config

    def gpt(self) -> Dict:
        if self.approach == "baseline":
            config = {"sleep": 10, "max_token_length": 500, "temperature": 0}
        else:
            config = {}
        return config

    def get_args(self, device="cpu"):
        self.parser.add_argument(
            "--root_dir",
            type=str,
            default=os.path.join(self.root_dataset_dir, "datasets"),
        )
        self.parser.add_argument(
            "--stats_dir",
            type=str,
            default=os.path.join(self.root_dataset_dir, "experiments", "stats"),
        )
        self.parser.add_argument("--FlanT5", type=dict, default=self.flan_t5(device))
        self.parser.add_argument("--LLaMA7B", type=dict, default=self.llama(device))
        self.parser.add_argument("--LLaMA13B", type=dict, default=self.llama(device))
        self.parser.add_argument("--Wizard13B", type=dict, default=self.llama(device))
        self.parser.add_argument("--ChatGPT", type=dict, default=self.gpt())
        self.parser.add_argument("--GPT4", type=dict, default=self.gpt())
        self.parser.add_argument("-f")
        return self.parser.parse_args()
