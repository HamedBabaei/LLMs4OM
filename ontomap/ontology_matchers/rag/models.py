# -*- coding: utf-8 -*-

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    MistralForCausalLM,
)

from ontomap.ontology_matchers.rag.rag import (
    RAG,
    RAGBasedDecoderLLMArch,
    RAGBasedOpenAILLMArch,
)
from ontomap.ontology_matchers.retrieval import AdaRetrieval, BERTRetrieval


class LLaMA7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = LlamaTokenizer
    model = LlamaForCausalLM
    path = "meta-llama/Llama-2-7b-hf"

    def __str__(self):
        return super().__str__() + "-LLaMA-2-7B"


class MistralLM7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = MistralForCausalLM
    path = "mistralai/Mistral-7B-v0.1"

    def __str__(self):
        return super().__str__() + "-MistralLM-7B-v0.1"


class ChatGPTOpenAILLM(RAGBasedOpenAILLMArch):
    path = "gpt-3.5-turbo-1106"

    def __str__(self):
        return super().__str__() + "-GPT-3.5"


class LLaMA7BLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = LLaMA7BDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA7BAdaRAG"


class MistralLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = MistralLM7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MistralLMAdaRAG"


class LLaMA7BLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = LLaMA7BDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA7BLLMBertRAG"


class MistralLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = MistralLM7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MistralLLMBertRAG"


class ChatGPTOpenAIAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = ChatGPTOpenAILLM

    def __str__(self):
        return super().__str__() + "-ChatGPTOpenAIAdaRAG"
