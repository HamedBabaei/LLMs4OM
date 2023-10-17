# -*- coding: utf-8 -*-
from typing import Any, List, Optional

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ontomap.base import BaseLLM


class EncoderDecoderLM(BaseLLM):
    def __init__(
        self,
        max_token_length: int,
        num_beams: Optional[int] = 10,
        device: Optional[str] = "cpu",
    ) -> None:
        super().__init__(
            max_token_length=max_token_length, num_beams=num_beams, device=device
        )

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        with torch.no_grad():
            sequence_ids = self.model.generate(
                tokenized_input_data.input_ids,
                num_beams=50,
                max_length=self.max_token_length,
                num_return_sequences=1,
            )
        sequences = self.tokenizer.batch_decode(
            sequence_ids.cpu(), skip_special_tokens=True
        )
        return sequences

    def generate_for_multiple_input(self, tokenized_input_data: Any) -> List:
        with torch.no_grad():
            sequence_ids = self.model.generate(
                input_ids=tokenized_input_data["input_ids"],
                attention_mask=tokenized_input_data["attention_mask"],
                max_new_tokens=self.max_token_length,
            )
        sequences = self.tokenizer.batch_decode(
            sequence_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return sequences


class FlanT5XXLEncoderDecoderLM(EncoderDecoderLM):
    tokenizer = T5Tokenizer
    model = T5ForConditionalGeneration
    path = "google/flan-t5-xxl"
