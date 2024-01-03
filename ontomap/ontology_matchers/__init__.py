# -*- coding: utf-8 -*-
from ontomap.ontology_matchers.icv import (
    FalconLLMAdaICV,
    FalconLLMBertICV,
    LLaMA7BLLMAdaICV,
    LLaMA7BLLMBertICV,
    MPTLLMAdaICV,
    MPTLLMBertICV,
    VicunaLLMAdaICV,
    VicunaLLMBertICV,
)
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
    FalconLLMAdaRAG,
    FalconLLMBertRAG,
    LLaMA7BLLMAdaRAG,
    LLaMA7BLLMBertRAG,
    MistralLLMAdaRAG,
    MistralLLMBertRAG,
    MPTLLMAdaRAG,
    MPTLLMBertRAG,
    VicunaLLMAdaRAG,
    VicunaLLMBertRAG,
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
        "FalconAdaRAG": FalconLLMAdaRAG,
        "FalconBertRAG": FalconLLMBertRAG,
        "VicunaBertRAG": VicunaLLMBertRAG,
        "VicunaAdaRAG": VicunaLLMAdaRAG,
        "MPTBertRAG": MPTLLMBertRAG,
        "MPTAdaRAG": MPTLLMAdaRAG,
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
    "icv": {
        "LLaMA7BAdaICV": LLaMA7BLLMAdaICV,
        "LLaMA7BBertICV": LLaMA7BLLMBertICV,
        "FalconAdaICV": FalconLLMAdaICV,
        "FalconBertICV": FalconLLMBertICV,
        "VicunaBertICV": VicunaLLMBertICV,
        "VicunaAdaICV": VicunaLLMAdaICV,
        "MPTBertICV": MPTLLMBertICV,
        "MPTAdaICV": MPTLLMAdaICV,
    },
}

__all__ = ["MatcherCatalog"]
