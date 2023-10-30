# -*- coding: utf-8 -*-
import time
from typing import Any, List

import openai
import torch

from ontomap.base import BaseLLM


class BaseLLMArch(BaseLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self):
        pass

    def load_model(self) -> None:
        if self.kwargs["device"] != "cpu":
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


class OpenAILLMArch(BaseLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self):
        return "OpenAILM"

    def load(self) -> None:
        pass

    def tokenize(self, input_data: List) -> Any:
        return input_data

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        prompt = [{"role": "user", "content": tokenized_input_data}]
        is_generated_output = False
        response = None
        while not is_generated_output:
            try:
                response = openai.ChatCompletion.create(
                    model=self.path,
                    messages=prompt,
                    temperature=self.kwargs["temperature"],
                    max_tokens=self.kwargs["max_token_length"],
                )
                is_generated_output = True
            except Exception as error:
                print(
                    f"Unexpected {error}, {type(error)} \n"
                    f"Going for sleep for {self.kwargs['sleep']} seconds!"
                )
                time.sleep(self.kwargs["sleep"])
        return [response]

    def generate_for_multiple_input(self, tokenized_input_data: Any) -> List:
        responses = []
        for input_data in tokenized_input_data:
            response = self.generate_for_one_input(tokenized_input_data=input_data)[0]
            responses.append(response)
        return responses

    def post_processor(self, generated_texts: List) -> List:
        processed_outputs = []
        for generated_text in generated_texts:
            processed_output = generated_text["choices"][0]["message"]["content"]
            processed_outputs.append(processed_output)
        return processed_outputs


class EncoderDecoderLLMArch(BaseLLMArch):
    def __str__(self):
        return "EncoderDecoderLM"


class DecoderLLMArch(BaseLLMArch):
    def __str__(self):
        return "DecoderLM"
