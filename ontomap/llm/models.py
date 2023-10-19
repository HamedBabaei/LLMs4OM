# -*- coding: utf-8 -*-
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from ontomap.llm.arch import DecoderLLMArch, EncoderDecoderLLMArch, OpenAILLMArch


class FlanT5XXLEncoderDecoderLM(EncoderDecoderLLMArch):
    tokenizer = T5Tokenizer
    model = T5ForConditionalGeneration
    path = "google/flan-t5-xxl"

    def __str__(self):
        return super().__str__() + "-FlanT5XXL"


class LLaMA7BDecoderLM(DecoderLLMArch):
    tokenizer = AutoTokenizer
    model = LlamaForCausalLM
    path = "meta-llama/Llama-2-7b-hf"

    def __str__(self):
        return super().__str__() + "-LLaMA-2-7B"


class LLaMA13BDecoderLM(DecoderLLMArch):
    tokenizer = AutoTokenizer
    model = LlamaForCausalLM
    path = "meta-llama/Llama-2-13b-hf"

    def __str__(self):
        return super().__str__() + "-LLaMA-2-13B"


class WizardLM13BDecoderLM(DecoderLLMArch):
    tokenizer = AutoTokenizer
    model = LlamaForCausalLM
    path = "WizardLM/WizardLM-13B-V1.2"

    def __str__(self):
        return super().__str__() + "-WizardLM-13B-V1.2"


class GPT4OpenAILLM(OpenAILLMArch):
    path = "gpt-4-0613"

    def __str__(self):
        return super().__str__() + "-GPT-4"


class ChatGPTOpenAILLM(OpenAILLMArch):
    path = "gpt-3.5-turbo-0613"

    def __str__(self):
        return super().__str__() + "-GPT-3.5"
