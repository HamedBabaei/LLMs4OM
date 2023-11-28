# -*- coding: utf-8 -*-
from typing import List

from tqdm import tqdm


def refactor_retrieval_predicts(predicts: List) -> List:
    predicts_temp = []
    predict_map = {}
    for predict in tqdm(predicts):
        source = predict["source"]
        target_cands = predict["target-cands"]
        score_cands = predict["score-cands"]
        for target, score in zip(target_cands, score_cands):
            if score > 0:
                adjusted = False
                # for predict_temp in predicts_temp:
                #     if (
                #         predict_temp["source"] == source
                #         and predict_temp["target"] == target
                #     ):
                if predict_map.get(f"{source}-{target}", "NA") != "NA":
                    adjusted = True
                    break
                if not adjusted:
                    predicts_temp.append(
                        {"source": source, "target": target, "score": score}
                    )
                    predict_map[f"{source}-{target}"] = f"{source}-{target}"
    return predicts_temp
