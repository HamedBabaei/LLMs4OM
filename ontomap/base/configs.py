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

    def get_args(self):
        self.parser.add_argument("--root_dir", type=str, default=self.root_dataset_dir)
        self.parser.add_argument("-f")
        return self.parser.parse_args()
