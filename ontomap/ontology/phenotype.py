# -*- coding: utf-8 -*-
import os.path
from typing import Any, List

from ontomap.base import BaseOntologyParser, OMDataset

track = "phenotype"


class DoidOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return owl_class.comment

    def get_label(self, owl_class: Any) -> str:
        return owl_class.label.first()

    def get_synonyms(self, owl_class: Any) -> List:
        return owl_class.hasExactSynonym


class OrdoOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return owl_class.definition

    def get_label(self, owl_class: Any) -> str:
        return owl_class.label.first()

    def get_synonyms(self, owl_class: Any) -> List:
        return []


class DoidOrdoOMDataset(OMDataset):
    track = track
    ontology_name = "doid-ordo"
    source_ontology = DoidOntology()
    target_ontology = OrdoOntology()
    working_dir = os.path.join(track, ontology_name)


class HpOntology(DoidOntology):
    pass


class MpOntology(DoidOntology):
    pass


class HpMpOMDataset(OMDataset):
    track = track
    ontology_name = "hp-mp"
    source_ontology = HpOntology()
    target_ontology = MpOntology()
    working_dir = os.path.join(track, ontology_name)
