# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from ontomap.evaluation.metrics import evaluation_report


def evaluator_module(track: str, predicts: List, references: Any) -> Dict:
    if track.startswith("bio-ml"):
        results = {
            "equiv": {
                "full": evaluation_report(
                    predicts=predicts, references=references["equiv"]["full"]
                ),
                "test": evaluation_report(
                    predicts=predicts, references=references["equiv"]["test"]
                ),
                "train": evaluation_report(
                    predicts=predicts, references=references["equiv"]["train"]
                ),
            },
            "subs": {
                "full": evaluation_report(
                    predicts=predicts,
                    references=(
                        references["subs"]["train"] + references["subs"]["test-cands"]
                    ),
                ),
                "test": evaluation_report(
                    predicts=predicts, references=references["subs"]["test-cands"]
                ),
                "train": evaluation_report(
                    predicts=predicts, references=references["subs"]["train"]
                ),
            },
        }
    elif track.startswith("bio-llm"):
        new_reference = [
            ref for ref in references["test-cands"] if ref["target"] != "UnMatched"
        ]
        results = evaluation_report(predicts=predicts, references=new_reference)
    else:
        results = evaluation_report(predicts=predicts, references=references)
    return results
