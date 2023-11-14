# -*- coding: utf-8 -*-
from typing import List

from ontomap.base import BaseOMModel


class Lightweight(BaseOMModel):
    path: str = "NO MODEL LOADING IN LIGHTWEIGHT MODEL"

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __str__(self):
        return "Lightweight"

    def generate(self, input_data: List) -> List:
        pass
