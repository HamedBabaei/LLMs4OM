# -*- coding: utf-8 -*-
import os.path
from typing import Any, List

from ontomap.base import BaseOntologyParser, OMDataset

track = "mse"


class EMMOOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return owl_class.comment.en

    def get_label(self, owl_class: Any) -> str:
        return owl_class.prefLabel.en


class MaterialInformationOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return []


class MatOntoOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return owl_class.comment

    def get_label(self, owl_class: Any) -> str:
        return owl_class.label

    def get_synonyms(self, owl_class: Any) -> List:
        return owl_class.synonym


class MaterialInformationEMMOOMDataset(OMDataset):
    track = track
    ontology_name = "MaterialInformation-EMMO"

    source_ontology = MaterialInformationOntology()
    target_ontology = EMMOOntology()

    working_dir = os.path.join(track, ontology_name)
