# -*- coding: utf-8 -*-
import os
from abc import ABC
from typing import Any, Dict, List

from owlready2 import World


class BaseOntologyParser(ABC):
    def __init__(self):
        pass

    def is_contain_label(self, owl_class: Any) -> bool:
        pass

    def get_name(self, owl_class: Any) -> str:
        return owl_class.name

    def get_label(self, owl_class: Any) -> str:
        return owl_class.label.first()

    def get_iri(self, owl_class: Any) -> str:
        return owl_class.iri

    def get_subclasses(self, owl_class: Any) -> List:
        pass

    def get_ancestors(self, owl_class: Any) -> List:
        pass

    def get_synonyms(self, owl_class: Any) -> List:
        pass

    def get_comments(self, owl_class: Any) -> List:
        pass

    def get_owl_items(self, owl_class: Any) -> List:
        owl_items = []
        for item in owl_class:
            if self.is_contain_label(item):
                owl_items.append(
                    {"iri": item.iri, "name": item.name, "label": item.label.first()}
                )
        return owl_items

    def extract_data(self, ontology: Any) -> List[Dict]:
        parsed_ontology = []
        for owl_class in ontology.classes():
            if not self.is_contain_label(owl_class):
                continue
            owl_class_info = {
                "name": self.get_name(owl_class),
                "iri": self.get_iri(owl_class),
                "label": self.get_label(owl_class),
                "subclasses": self.get_subclasses(owl_class),
                "ancestors": self.get_ancestors(owl_class),
                "synonyms": self.get_synonyms(owl_class),
                "comment": self.get_comments(owl_class),
            }
            parsed_ontology.append(owl_class_info)
        return parsed_ontology

    @staticmethod
    def load_ontology(input_file_path: str) -> World:
        ontology = World()
        ontology.get_ontology(input_file_path).load()
        return ontology

    def parse(self, root_dir: str, ontology_file_name: str) -> List:
        input_file_path = os.path.join(root_dir, ontology_file_name)
        ontology = self.load_ontology(input_file_path=input_file_path)
        return self.extract_data(ontology)
