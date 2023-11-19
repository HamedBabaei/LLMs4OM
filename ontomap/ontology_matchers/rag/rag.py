# -*- coding: utf-8 -*-
from typing import List

from ontomap.base import BaseOMModel


class RAG(BaseOMModel):
    path: str = "NO MODEL LOADING IN LIGHTWEIGHT MODEL"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # define supper class loadings here
        # define LLM loading also here - this can be similar to LLM module -

    def __str__(self):
        return "RAG"

    def generate(self, input_data: List) -> List:
        pass
