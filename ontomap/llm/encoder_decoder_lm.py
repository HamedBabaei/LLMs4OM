# -*- coding: utf-8 -*-
from typing import Any, List, Optional

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ontomap.base import BaseLLM


class FlanT5EncoderDecoderLM(BaseLLM):
    tokenizer = T5Tokenizer
    model = T5ForConditionalGeneration

    def __init__(self, path: str, device: Optional[str] = "cpu") -> None:
        super().__init__(path=path, device=device)

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        with torch.no_grad():
            sequence_ids = self.model.generate(
                tokenized_input_data.input_ids, num_beams=50, max_length=5
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
                max_new_tokens=5,
            )
        sequences = self.tokenizer.batch_decode(
            sequence_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return sequences
