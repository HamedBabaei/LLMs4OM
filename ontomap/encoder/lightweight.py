# -*- coding: utf-8 -*-
from typing import Any, Dict

from ontomap.encoder.encoders import LightweightEncoder


class IRILabelInLightweightEncoder(LightweightEncoder):
    items_in_owl: str = """(Label)"""

    def get_owl_items(self, owl: Dict) -> Any:
        return {"iri": owl["iri"], "text": owl["label"]}


class IRILabelDescInLightweightEncoder(LightweightEncoder):
    items_in_owl: str = "(Label, Description)"

    def get_owl_items(self, owl: Dict) -> Any:
        return {"iri": owl["iri"], "text": owl["label"] + ", " + str(owl["comment"])}


class IRILabelChildrensInLightweightEncoder(LightweightEncoder):
    items_in_owl: str = "(Label, Childrens)"

    def get_owl_items(self, owl: Dict) -> Any:
        childrens = ", ".join([children["label"] for children in owl["childrens"]])
        return {"iri": owl["iri"], "text": owl["label"] + "  " + str(childrens)}


class IRILabelParentsInLightweightEncoder(LightweightEncoder):
    items_in_owl: str = "(Label, Parents)"

    def get_owl_items(self, owl: Dict) -> Any:
        parents = ", ".join([parent["label"] for parent in owl["parents"]])
        return {"iri": owl["iri"], "text": owl["label"] + "  " + str(parents)}
