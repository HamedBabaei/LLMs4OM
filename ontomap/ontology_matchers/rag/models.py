# -*- coding: utf-8 -*-

from transformers import (
    AutoModelForCausalLM,
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
from ontomap.ontology_matchers.retrieval.models import AdaRetrieval, BERTRetrieval


class LLaMA7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = LlamaTokenizer
    model = LlamaForCausalLM
    path = "meta-llama/Llama-2-7b-hf"

    def __str__(self):
        return super().__str__() + "-LLaMA-2-7B"


class Mistral7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = MistralForCausalLM
    path = "mistralai/Mistral-7B-v0.1"

    def __str__(self):
        return super().__str__() + "-MistralLM-7B-v0.1"


class Falcon7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "tiiuae/falcon-7b"

    def __str__(self):
        return super().__str__() + "-falcon-7b"

    def get_probas_yes_no(self, outputs):
        probas_yes_no = (
            outputs.scores[0][:, self.answer_sets_token_id["yes"] + self.answer_sets_token_id["no"]]
            .float()
            .softmax(-1)
        )
        return probas_yes_no

    def check_answer_set_tokenizer(self, answer: str) -> bool:
        return len(self.tokenizer(answer).input_ids) == 1


class Vicuna7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "lmsys/vicuna-7b-v1.5"

    def __str__(self):
        return super().__str__() + "-vicuna-7b-v1.5"


class MPT7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "mosaicml/mpt-7b"

    def __str__(self):
        return super().__str__() + "-mpt-7b"

    def get_probas_yes_no(self, outputs):
        probas_yes_no = (
            outputs.scores[0][:, self.answer_sets_token_id["yes"] + self.answer_sets_token_id["no"]]
            .float()
            .softmax(-1)
        )
        return probas_yes_no

    def check_answer_set_tokenizer(self, answer: str) -> bool:
        return len(self.tokenizer(answer).input_ids) == 1


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
    LLM = Mistral7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MistralLMAdaRAG"


class LLaMA7BLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = LLaMA7BDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA7BLLMBertRAG"


class MistralLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = Mistral7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MistralLLMBertRAG"


class ChatGPTOpenAIAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = ChatGPTOpenAILLM

    def __str__(self):
        return super().__str__() + "-ChatGPTOpenAIAdaRAG"


class FalconLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = Falcon7BDecoderLM

    def __str__(self):
        return super().__str__() + "-FalconLLMAdaRAG"


class FalconLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = Falcon7BDecoderLM

    def __str__(self):
        return super().__str__() + "-FalconLLMBertaRAG"


class VicunaLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = Vicuna7BDecoderLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMAdaRAG"


class VicunaLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = Vicuna7BDecoderLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMBertRAG"


class MPTLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = MPT7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MPTLLMAdaRAG"


class MPTLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = MPT7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MPTLLMBertRAG"
