# -*- coding: utf-8 -*-
import os
from abc import ABC
from typing import Any, Dict

from ontomap.base import BaseAlignmentsParser


class OMDataset(ABC):
    track: str = ""
    ontology_name: str = ""

    source_ontology: Any = None
    target_ontology: Any = None

    alignments: Any = BaseAlignmentsParser()

    working_dir: str = ""

    def collect(self, root_dir: str) -> Dict:
        om_root_path = os.path.join(root_dir, self.track, self.ontology_name)
        data = {
            "dataset-info": {"track": self.track, "ontology-name": self.ontology_name},
            "source": self.source_ontology.parse(
                root_dir=om_root_path, ontology_file_name="source.xml"
            ),
            "target": self.target_ontology.parse(
                root_dir=om_root_path, ontology_file_name="target.xml"
            ),
            "reference": self.alignments.parse(
                root_dir=om_root_path, reference_file_name="reference.xml"
            ),
        }
        return data

    def __dir__(self):
        return os.path.join(self.track, self.ontology_name)
