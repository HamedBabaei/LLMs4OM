# -*- coding: utf-8 -*-
import os.path
from typing import Any, List

from ontomap.base import BaseOntologyParser, OMDataset

track = "largebio"


class BioOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.comment)

    def get_synonyms(self, owl_class: Any) -> List:
        return []


class FMANCIOMDataset(OMDataset):
    track = track
    ontology_name = "fma-nci"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)


class FMASNOMEDOMDataset(OMDataset):
    track = track
    ontology_name = "fma-snomed"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)


class SNOMEDNCIOMDataset(OMDataset):
    track = track
    ontology_name = "snomed-nci"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
