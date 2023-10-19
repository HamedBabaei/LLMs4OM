# -*- coding: utf-8 -*-
"""
    BaseConfig: Data Configuration of models
"""
import argparse
import os
from typing import Optional


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

    def flant5(self, device: str):
        if self.approach == "baseline":
            config = {"max_token_length": 500, "num_beams": 10, "device": device}
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
        self.parser.add_argument(
            "--FlanT5",
            type=dict,
            default=self.flant5(device),
        )
        self.parser.add_argument("-f")
        return self.parser.parse_args()
