# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from ontomap.base import BaseEncoder


class LightweightEncoder(BaseEncoder):
    items_in_owl: str = ""

    def parse(self, **kwargs) -> List:
        source_onto, target_onto = kwargs["source"], kwargs["target"]
        source_ontos = []
        for source in source_onto:
            source_ontos.append(self.get_owl_items(owl=source))
        target_ontos = []
        for target in target_onto:
            target_ontos.append(self.get_owl_items(owl=target))
        return [source_ontos, target_ontos]

    def __str__(self):
        return {"LightweightEncoder": self.items_in_owl}

    def get_owl_items(self, owl: Dict) -> Any:
        pass

    def get_encoder_info(self):
        return "INPUT CONSIST OF COMBINED INFORMATION TO FUZZY STRING MATCHING"


class IRILabelInLightweightEncoder(LightweightEncoder):
    items_in_prompt: str = "(Label)"

    def get_owl_items(self, owl: Dict) -> Any:
        return {"iri": owl["iri"], "text": owl["label"]}


class IRILabelDescInLightweightEncoder(LightweightEncoder):
    items_in_prompt: str = "(Label, Description)"

    def get_owl_items(self, owl: Dict) -> Any:
        return {"iri": owl["iri"], "text": owl["label"] + "  " + str(owl["comment"])}


class IRILabelChildrensInLightweightEncoder(LightweightEncoder):
    items_in_prompt: str = "(Label, Childrens)"

    def get_owl_items(self, owl: Dict) -> Any:
        childrens = " ".join([children["label"] for children in owl["childrens"]])
        return {"iri": owl["iri"], "text": owl["label"] + "  " + str(childrens)}


class IRILabelParentsInLightweightEncoder(LightweightEncoder):
    items_in_prompt: str = "(Label, Parents)"

    def get_owl_items(self, owl: Dict) -> Any:
        parents = " ".join([parent["label"] for parent in owl["parents"]])
        return {"iri": owl["iri"], "text": owl["label"] + "  " + str(parents)}
