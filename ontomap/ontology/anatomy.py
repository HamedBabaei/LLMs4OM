# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from ontomap.common.base import BaseOntologyParser

track = "anatomy"


class MouseOntology(BaseOntologyParser):
    def __init__(self):
        super().__init__()

    def is_contain_label(self, owl_class: Any) -> bool:
        if len(owl_class.label) == 0:
            return False
        return True

    def get_label(self, owl_class: Any) -> str:
        return owl_class.label.first()

    def get_subclasses(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.subclasses())

    def get_ancestors(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.ancestors())

    def get_synonyms(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.hasRelatedSynonym)

    def get_comments(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.comment)

    def parse(self, root_dir: str, ontology_file_name: str = "source.xml") -> List:
        return super().parse(root_dir=root_dir, ontology_file_name=ontology_file_name)


class HumanOntology(MouseOntology):
    def parse(self, root_dir: str, ontology_file_name: str = "target.xml") -> List:
        return super().parse(root_dir=root_dir, ontology_file_name=ontology_file_name)


class MouseHumanOMDataset:
    track = track
    ontology_name = "mouse-human"

    source_ontology = MouseOntology()
    target_ontology = HumanOntology()

    def get_data(self, root_dir: str) -> Dict:
        data = {
            "source": self.source_ontology.parse(root_dir=root_dir),
            "target": self.target_ontology.parse(root_dir=root_dir),
        }
        return data
