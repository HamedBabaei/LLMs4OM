# -*- coding: utf-8 -*-
from ontomap.llm.models import (
    ChatGPTOpenAILLM,
    FlanT5XXLEncoderDecoderLM,
    GPT4OpenAILLM,
    LLaMA7BDecoderLM,
    LLaMA13BDecoderLM,
    WizardLM13BDecoderLM,
)

LLMCatalog = {
    "FlanT5": FlanT5XXLEncoderDecoderLM,
    "LLaMA7B": LLaMA7BDecoderLM,
    "LLaMA13B": LLaMA13BDecoderLM,
    "Wizard13B": WizardLM13BDecoderLM,
    "ChatGPT": ChatGPTOpenAILLM,
    "GPT4": GPT4OpenAILLM,
}

__all__ = ["LLMCatalog"]
