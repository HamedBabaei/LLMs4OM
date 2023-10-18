# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class BasePrompt(ABC):
    prompt_template: str = ""

    def __str__(self):
        return self.prompt_template

    @abstractmethod
    def parse(self, **kwargs) -> str:
        pass

    def __call__(self, **kwargs):
        return self.parse(**kwargs)
