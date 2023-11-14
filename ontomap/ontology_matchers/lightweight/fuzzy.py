# -*- coding: utf-8 -*-
from typing import List

import rapidfuzz
from tqdm import tqdm

from ontomap.ontology_matchers.lightweight.lightweight import Lightweight


class FuzzySMLightweight(Lightweight):
    """
    Fuzzy String Matching using: https://github.com/maxbachmann/RapidFuzz#partial-ratio
    """

    def __str__(self):
        return super().__str__() + "+FuzzySM"

    def generate(self, input_data: List) -> List:
        source_ontology = input_data[0]
        target_ontology = input_data[1]
        predictions = []
        candidates = [target["text"] for target in target_ontology]
        for source in tqdm(source_ontology):
            selected_candid = rapidfuzz.process_cpp.extractOne(
                source["text"],
                candidates,
                scorer=rapidfuzz.fuzz.ratio,
                processor=rapidfuzz.utils.default_process,
            )
            if selected_candid[1] / 100 >= self.kwargs["fuzzy_sm_threshold"]:
                predictions.append(
                    {
                        "source": source["iri"],
                        "target": target_ontology[selected_candid[2]]["iri"],
                        "score": selected_candid[1] / 100,
                    }
                )
        return predictions
