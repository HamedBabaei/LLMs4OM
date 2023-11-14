# -*- coding: utf-8 -*-
from typing import Any, List

from rapidfuzz import fuzz

from ontomap.ontology_matchers.lightweight.lightweight import Lightweight


class FuzzySMLightweight(Lightweight):
    def __str__(self):
        return super().__str__() + "+FuzzySM"

    def generate(self, input_data: List) -> List:
        source_ontology = input_data[0]
        target_ontology = input_data[1]
        predictions = []
        for source in source_ontology:
            for target in target_ontology:
                ratio = fuzz.ratio(source["text"], target["text"]) / 100
                if ratio >= self.kwargs["fuzzy_sm_threshold"]:
                    predictions.append(
                        {
                            "source": source["iri"],
                            "target": target["iri"],
                            "score": ratio,
                        }
                    )
        return predictions
