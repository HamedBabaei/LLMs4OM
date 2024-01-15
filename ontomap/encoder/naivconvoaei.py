# -*- coding: utf-8 -*-
from typing import Dict

from ontomap.encoder.encoders import NaiveConvOAEIEncoder


class IRILabelInNaiveEncoder(NaiveConvOAEIEncoder):
    items_in_owl: str = "(IRI, Label)"

    def get_owl_items(self, owl: Dict) -> str:
        return f"({owl['iri']}, {owl['label']}), "


class IRILabelDescInNaiveEncoder(NaiveConvOAEIEncoder):
    items_in_owl: str = "(IRI, Label, Description)"

    def get_owl_items(self, owl: Dict) -> str:
        return f"({owl['iri']}, {owl['label']}, {str(owl['comment'])}), "


class IRILabelChildrensInNaiveEncoder(NaiveConvOAEIEncoder):
    items_in_owl: str = "(IRI, Label, Childrens)"

    def get_owl_items(self, owl: Dict) -> str:
        childrens = [children["label"] for children in owl["childrens"]]
        return f"({owl['iri']}, {owl['label']}, {str(childrens)}), "


class IRILabelParentsInNaiveEncoder(NaiveConvOAEIEncoder):
    items_in_owl: str = "(IRI, Label, Parents)"

    def get_owl_items(self, owl: Dict) -> str:
        parents = [parent["label"] for parent in owl["parents"]]
        return f"({owl['iri']}, {owl['label']}, {str(parents)}), "
