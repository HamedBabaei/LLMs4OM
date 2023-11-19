# -*- coding: utf-8 -*-
from ontomap.ontology_matchers.lightweight import (
    SimpleFuzzySMLightweight,
    TokenSetFuzzySMLightweight,
    WeightedFuzzySMLightweight,
)
from ontomap.ontology_matchers.llm import (
    ChatGPTOpenAILLM,
    FlanT5XLEncoderDecoderLM,
    FlanT5XXLEncoderDecoderLM,
    GPT4OpenAILLM,
    LLaMA7BDecoderLM,
    LLaMA13BDecoderLM,
    MistralLM7BDecoderLM,
    WizardLM13BDecoderLM,
)
from ontomap.ontology_matchers.retrieval import (
    BERTRetrieval,
    BM25Retrieval,
    FlanT5XLRetrieval,
    FlanT5XXLRetrieval,
    SpecterBERTRetrieval,
    TFIDFRetrieval,
)

MatcherCatalog = {
    "naiv-conv-oaei": {
        "FlanT5XL": FlanT5XLEncoderDecoderLM,
        "FlanT5XXL": FlanT5XXLEncoderDecoderLM,
        "LLaMA7B": LLaMA7BDecoderLM,
        "LLaMA13B": LLaMA13BDecoderLM,
        "Wizard13B": WizardLM13BDecoderLM,
        "Mistral7B": MistralLM7BDecoderLM,
        "ChatGPT": ChatGPTOpenAILLM,
        "GPT4": GPT4OpenAILLM,
    },
    "lightweight": {
        "SimpleFuzzySM": SimpleFuzzySMLightweight,
        "WeightedFuzzySM": WeightedFuzzySMLightweight,
        "TokenSetFuzzySM": TokenSetFuzzySMLightweight,
    },
    "rag": {
        "BERTLLaMARAG": 1,
        "BERTMistralRAG": 2,
        "BERTGPT4RAG": 3,
        "FlanT5LLaMARAG": 4,
        "FlanT5MistralRAG": 5,
        "FlanT5GPT4RAG": 6,
    },
    "retrieval": {
        "BM25Retrieval": BM25Retrieval,
        "TFIDFRetrieval": TFIDFRetrieval,
        "BERTRetrieval": BERTRetrieval,
        "SpecterBERTRetrieval": SpecterBERTRetrieval,
        "FlanT5XLRetrieval": FlanT5XLRetrieval,
        "FlanT5XXRetrieval": FlanT5XXLRetrieval,
    },
}

__all__ = ["MatcherCatalog"]
