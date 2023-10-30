# -*- coding: utf-8 -*-
"""
    BaseConfig: Data Configuration of models
"""
import argparse
import os
import sys
from typing import Dict, Optional

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())


class BaseConfig:
    def __init__(self, approach: Optional[str] = "none"):
        self.root_dataset_dir = sys.path[1]
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
        if self.approach == "out-of-box":
            config = {"max_token_length": 5000, "num_beams": 10, "device": device}
        else:
            config = {}
        return config

    def llama(self, device: str) -> Dict:
        if self.approach == "out-of-box":
            config = {
                "max_token_length": 5000,
                "num_beams": 10,
                "device": device,
                "HUGGINGFACE_ACCESS_TOKEN": os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            }
        else:
            config = {}
        return config

    def gpt(self) -> Dict:
        if self.approach == "out-of-box":
            config = {"sleep": 10, "max_token_length": 5000, "temperature": 0}
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
            "--output_dir",
            type=str,
            default=os.path.join(self.root_dataset_dir, "experiments", "outputs"),
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
