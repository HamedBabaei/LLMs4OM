# -*- coding: utf-8 -*-
"""
    BaseConfig: Data Configuration of models
"""
import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import openai
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())


class BaseConfig:
    def __init__(self, approach: Optional[str] = "none"):
        self.root_dataset_dir = Path(__file__).parents[2]
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
        config = {
            "max_token_length": 5000,
            "truncation": False,
            "num_beams": 1,
            "device": device,
            "top_p": 0.95,
            "temperature": 0.8,
        }
        if self.approach == "naiv-conv-oaei":
            config["tokenizer_max_length"] = 4096
        else:
            config = {}
        return config

    def llama(self, device: str) -> Dict:
        config = {
            "max_token_length": 5000,
            "num_beams": 1,
            "device": device,
            "truncation": False,
            "HUGGINGFACE_ACCESS_TOKEN": os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            "top_p": 0.95,
            "temperature": 0.8,
        }
        if self.approach == "naiv-conv-oaei":
            config["tokenizer_max_length"] = 4096
        else:
            config = {}
        return config

    def gpt(self) -> Dict:
        openai.api_key = os.environ["OPENAI_KEY"]
        # due to the privacy I had to set key here,
        # but the correct way is to set inside OPENAILLMARCH class
        config = {
            "sleep": 10,
            "top_p": 0.95,
            "temperature": 0.8,
        }
        if self.approach == "naiv-conv-oaei":
            config["max_token_length"] = 5000
        else:
            config = {}
        return config

    def fuzzy(self) -> Dict:
        config = {"fuzzy_sm_threshold": 0.8}
        return config

    def retrieval(self, device: str) -> Dict:
        config = {
            "top_k": 5,
            "device": device,
        }
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
        # LLM configuration
        flan_t5_config = self.flan_t5(device)
        llama_config = self.llama(device)
        gpt_config = self.gpt()
        self.parser.add_argument("--FlanT5", type=dict, default=flan_t5_config)
        self.parser.add_argument("--LLaMA7B", type=dict, default=llama_config)
        self.parser.add_argument("--LLaMA13B", type=dict, default=llama_config)
        self.parser.add_argument("--Wizard13B", type=dict, default=llama_config)
        self.parser.add_argument("--Mistral7B", type=dict, default=llama_config)
        self.parser.add_argument("--ChatGPT", type=dict, default=gpt_config)
        self.parser.add_argument("--GPT4", type=dict, default=gpt_config)

        # Lightweight Configuration
        fuzzy_config = self.fuzzy()
        self.parser.add_argument("--SimpleFuzzySM", type=dict, default=fuzzy_config)
        self.parser.add_argument("--WeightedFuzzySM", type=dict, default=fuzzy_config)
        self.parser.add_argument("--TokenSetFuzzySM", type=dict, default=fuzzy_config)

        # Retrieval Configurations
        retriever_config = self.retrieval(device)
        self.parser.add_argument(
            "--TFIDFRetrieval", type=dict, default=retriever_config
        )
        self.parser.add_argument("--BERTRetrieval", type=dict, default=retriever_config)
        self.parser.add_argument(
            "--SpecterBERTRetrieval", type=dict, default=retriever_config
        )
        self.parser.add_argument("--BM25Retrieval", type=dict, default=retriever_config)
        self.parser.add_argument(
            "--FlanT5XLRetrieval", type=dict, default=retriever_config
        )
        self.parser.add_argument(
            "--FlanT5XXLRetrieval", type=dict, default=retriever_config
        )
        rag_config = {
            "retriever-config": retriever_config,
            "llm-config": flan_t5_config,
        }
        self.parser.add_argument("--RAG", type=dict, default=rag_config)
        # RAG configurations

        self.parser.add_argument("-f")
        return self.parser.parse_args()
