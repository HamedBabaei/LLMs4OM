# -*- coding: utf-8 -*-
from abc import ABC
from typing import Dict


class OMDataset(ABC):
    track: str = None
    ontology_name: str = None

    source_ontology = None
    target_ontology = None

    def get_data(self, root_dir: str) -> Dict:
        pass
