# -*- coding: utf-8 -*-
from typing import Any, List

import torch

from ontomap.base import BaseLLM


class BaseLLMArch(BaseLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self):
        pass

    def load_model(self) -> None:
        if self.kwargs["device"] == "gpu":
            self.model = self.model.from_pretrained(self.path, device_map="balanced")
        else:
            super().load_model()

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        with torch.no_grad():
            sequence_ids = self.model.generate(
                tokenized_input_data.input_ids,
                num_beams=self.kwargs["num_beams"],
                max_length=self.kwargs["max_token_length"],
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
                max_new_tokens=self.kwargs["max_token_length"],
            )
        sequences = self.tokenizer.batch_decode(
            sequence_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return sequences


class EncoderDecoderLLMArch(BaseLLMArch):
    def __str__(self):
        return "EncoderDecoderLM"


class DecoderLLMArch(BaseLLMArch):
    def __str__(self):
        return "DecoderLM"
