# -*- coding: utf-8 -*-
"""
    BaseConfig: Data Configuration of models
"""
import argparse
import os


class BaseConfig:
    def __init__(self, root_dataset_dir):
        self.root_dataset_dir = root_dataset_dir
        self.parser = argparse.ArgumentParser()

    def mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def add_path(self):
        pass

    def __str__(self):
        return "config."

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
            default={"max_token_length": 500, "num_beams": 10, "device": device},
        )
        self.parser.add_argument("-f")
        return self.parser.parse_args()
