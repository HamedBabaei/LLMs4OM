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
from ontomap.ontology_matchers.rag import (
    ChatGPTOpenAIAdaRAG,
    LLaMA7BLLMAdaRAG,
    LLaMA7BLLMBertRAG,
    MistralLLMAdaRAG,
    MistralLLMBertRAG,
)
from ontomap.ontology_matchers.retrieval import (
    AdaRetrieval,
    BERTRetrieval,
    BM25Retrieval,
    FlanT5XLRetrieval,
    FlanT5XXLRetrieval,
    SpecterBERTRetrieval,
    SVMBERTRetrieval,
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
        "LLaMA7BAdaRAG": LLaMA7BLLMAdaRAG,
        "MistralAdaRAG": MistralLLMAdaRAG,
        "ChatGPTOpenAIAdaRAG": ChatGPTOpenAIAdaRAG,
        "LLaMA7BBertRAG": LLaMA7BLLMBertRAG,
        "MistralBertRAG": MistralLLMBertRAG,
    },
    "retrieval": {
        "BM25Retrieval": BM25Retrieval,
        "TFIDFRetrieval": TFIDFRetrieval,
        "BERTRetrieval": BERTRetrieval,
        "SpecterBERTRetrieval": SpecterBERTRetrieval,
        "FlanT5XLRetrieval": FlanT5XLRetrieval,
        "FlanT5XXLRetrieval": FlanT5XXLRetrieval,
        "SVMBERTRetrieval": SVMBERTRetrieval,
        "AdaRetrieval": AdaRetrieval,
    },
}

__all__ = ["MatcherCatalog"]
