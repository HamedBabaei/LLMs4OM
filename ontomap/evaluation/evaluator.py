# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from ontomap.evaluation.metrics import evaluation_report


def evaluator(track: str, predicts: List, references: Any):
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


def refactor_retrieval_predicts(predicts: List) -> List:
    predicts_temp = []
    for predict in predicts:
        source = predict["source"]
        target_cands = predict["target-cands"]
        score_cands = predict["score-cands"]
        for target, score in zip(target_cands, score_cands):
            if score > 0:
                predicts_temp.append(
                    {"source": source, "target": target, "score": score}
                )
    return predicts_temp


def evaluator_module(
    track: str, approach: str, predicts: List, references: Any
) -> Dict:
    if approach == "retrieval":
        predicts = refactor_retrieval_predicts(predicts=predicts)
    results = evaluator(track=track, predicts=predicts, references=references)
    return results
