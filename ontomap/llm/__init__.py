# -*- coding: utf-8 -*-
from ontomap.llm.models import (
    FlanT5XXLEncoderDecoderLM,
    LLaMA7BDecoderLM,
    LLaMA13BDecoderLM,
    WizardLM13BDecoderLM,
)

LLMCatalog = {
    "FlanT5": FlanT5XXLEncoderDecoderLM,
    "LLaMA-2-7B": LLaMA7BDecoderLM,
    "LLaMA-2-13B": LLaMA13BDecoderLM,
    "Wizard-2-13B": WizardLM13BDecoderLM,
}

__all__ = ["LLMCatalog"]
