# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseLLM(ABC):
    tokenizer: Any = None
    model: Any = None
    path: str = ""

    def __init__(
        self,
        max_token_length: int,
        num_beams: Optional[int] = 10,
        device: Optional[str] = "cpu",
    ) -> None:
        self.device = device
        self.max_token_length = max_token_length
        self.num_beans = num_beams
        self.load()

    @abstractmethod
    def __str__(self):
        pass

    def load(self) -> None:
        self.load_tokenizer()
        self.load_model()

    def load_tokenizer(self) -> None:
        self.tokenizer = self.tokenizer.from_pretrained(self.path)

    def load_model(self) -> None:
        self.model = self.model.from_pretrained(self.path)
        self.model.to(self.device)

    def tokenize(self, input_data: List) -> Any:
        inputs = self.tokenizer(input_data, return_tensors="pt", padding=True)
        inputs.to(self.device)
        return inputs

    def generate(self, input_data: List) -> List:
        tokenized_input_data = self.tokenize(input_data=input_data)
        if len(input_data) == 1:
            generated_texts = self.generate_for_one_input(
                tokenized_input_data=tokenized_input_data
            )
        else:
            generated_texts = self.generate_for_multiple_input(
                tokenized_input_data=tokenized_input_data
            )
        generated_texts = self.post_processor(generated_texts=generated_texts)
        return generated_texts

    @abstractmethod
    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        pass

    @abstractmethod
    def generate_for_multiple_input(self, tokenized_input_data: Any) -> List:
        pass

    def post_processor(self, generated_texts: List) -> List:
        return generated_texts
