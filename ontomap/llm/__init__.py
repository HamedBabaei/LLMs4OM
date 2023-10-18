# -*- coding: utf-8 -*-
from ontomap.llm.encoder_decoder_lm import FlanT5XXLEncoderDecoderLM

LLMCatalog = {"FlanT5": [FlanT5XXLEncoderDecoderLM]}

__all__ = ["LLMCatalog"]
