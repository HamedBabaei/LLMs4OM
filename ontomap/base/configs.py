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
        self.batch_size = None
        self.device = None
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

    def flan_t5(self) -> Dict:
        config = {
            "max_token_length": 5000,
            "truncation": False,
            "num_beams": 1,
            "device": self.device,
            "top_p": 0.95,
            "temperature": 0.8,
            "padding": "max_length",
            "batch_size": self.batch_size,
        }
        if self.approach == "naiv-conv-oaei":
            config["tokenizer_max_length"] = 4096
        elif self.approach == "rag":
            config["tokenizer_max_length"] = 300
            config["max_token_length"] = 1
        else:
            pass
        return config

    def llama(self) -> Dict:
        config = {
            "max_token_length": 5000,
            "num_beams": 1,
            "device": self.device,
            "truncation": True,
            # "HUGGINGFACE_ACCESS_TOKEN": os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            "top_p": 0.95,
            "temperature": 0.8,
            "batch_size": self.batch_size,
            "padding": "max_length",
        }
        if self.approach == "naiv-conv-oaei":
            config["tokenizer_max_length"] = 4096
        elif self.approach == "rag":
            config["tokenizer_max_length"] = 500
            config["max_token_length"] = 1
        else:
            pass

        return config

    def gpt(self) -> Dict:
        openai.api_key = os.environ["OPENAI_KEY"]
        # due to the privacy I had to set key here,
        # but the correct way is to set inside OPENAILLMARCH class
        config = {
            "sleep": 5,
            "top_p": 0.95,
            "temperature": 0,
            "batch_size": self.batch_size,
        }
        if self.approach == "naiv-conv-oaei":
            config["max_token_length"] = 5000
        elif self.approach == "rag":
            config["max_token_length"] = 2
        else:
            pass
        return config

    def fuzzy(self) -> Dict:
        config = {"fuzzy_sm_threshold": 0.8}
        return config

    def retrieval(self) -> Dict:
        config = {
            "top_k": 5,
            "device": self.device,
        }
        return config

    def get_args(self, device="cpu", batch_size: int = None):
        self.device = device
        self.batch_size = batch_size
        self.parser.add_argument(
            "--root_dir",
            type=str,
            default=os.path.join(self.root_dataset_dir, "datasets"),
        )
        self.parser.add_argument(
            "--experiments_dir",
            type=str,
            default=os.path.join(self.root_dataset_dir, "experiments"),
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
        flan_t5_config, llama_config, gpt_config = (
            self.flan_t5(),
            self.llama(),
            self.gpt(),
        )
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
        retriever_config = self.retrieval()
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
        self.parser.add_argument(
            "--SVMBERTRetrieval", type=dict, default=retriever_config
        )
        self.parser.add_argument("--AdaRetrieval", type=dict, default=retriever_config)
        self.parser.add_argument(
            "--openai_embedding_dir",
            type=str,
            default=os.path.join(self.root_dataset_dir, "assets", "openai-embedding"),
        )

        llama_rag_config = {
            "retriever-config": retriever_config,
            "llm-config": llama_config,
        }
        self.parser.add_argument("--LLaMA7BAdaRAG", type=dict, default=llama_rag_config)
        self.parser.add_argument("--MistralAdaRAG", type=dict, default=llama_rag_config)
        self.parser.add_argument(
            "--LLaMA7BBertRAG", type=dict, default=llama_rag_config
        )
        self.parser.add_argument(
            "--MistralBertRAG", type=dict, default=llama_rag_config
        )

        openai_rag_config = {
            "retriever-config": retriever_config,
            "llm-config": gpt_config,
        }
        self.parser.add_argument(
            "--ChatGPTOpenAIAdaRAG", type=dict, default=openai_rag_config
        )
        # RAG configurations

        self.parser.add_argument("-f")
        return self.parser.parse_args()
