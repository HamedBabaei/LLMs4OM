# -*- coding: utf-8 -*-
from typing import Dict, List

from tqdm import tqdm


def eval_preprocess_ir_outputs(predicts: List) -> List:
    predicts_temp = []
    predict_map = {}
    for predict in tqdm(predicts):
        source = predict["source"]
        target_cands = predict["target-cands"]
        score_cands = predict["score-cands"]
        for target, score in zip(target_cands, score_cands):
            if score > 0:
                adjusted = False
                if predict_map.get(f"{source}-{target}", "NA") != "NA":
                    adjusted = True
                    break
                if not adjusted:
                    predicts_temp.append(
                        {"source": source, "target": target, "score": score}
                    )
                    predict_map[f"{source}-{target}"] = f"{source}-{target}"
    return predicts_temp


def preprocess_ir_outputs(predicts: List) -> List:
    predicts_temp = []
    for predict in tqdm(predicts):
        source, target_cands, score_cands = (
            predict["source"],
            predict["target-cands"],
            predict["score-cands"],
        )
        for target, score in zip(target_cands, score_cands):
            if score > 0:
                predicts_temp.append(
                    {"source": source, "target": target, "score": score}
                )
    return predicts_temp


def threshold_finder(dictionary: dict, index: int, use_lst: bool = False) -> float:
    scores_dict = {}
    scores_list = []
    for outputs in dictionary.values():
        for output in outputs:
            scores_list.append(output[index])
            if scores_dict.get(output[0], 0) != 0:
                if scores_dict.get(output[0], 0) < output[index]:
                    scores_dict[output[0]] = output[index]
            else:
                scores_dict[output[0]] = output[index]
    if not use_lst:
        scores_list = list(scores_dict.values())
    threshold = sum(scores_list) / len(scores_list)
    return threshold


def build_outputdict(llm_outputs: List, ir_outputs: List) -> Dict:
    outputdict = {}
    for llm_output in tqdm(llm_outputs):
        for ir_output in ir_outputs:
            if (
                llm_output["source"] == ir_output["source"]
                and llm_output["target"] == ir_output["target"]
            ):
                confidence_ratio = llm_output["score"] * ir_output["score"]
                predicts_list = [
                    llm_output["target"],
                    ir_output["score"],
                    llm_output["score"],
                    confidence_ratio,
                ]
                if llm_output["source"] not in list(outputdict.keys()):
                    outputdict[llm_output["source"]] = [predicts_list]
                else:
                    outputdict[llm_output["source"]].append(predicts_list)
    return outputdict


def confidence_score_ratio_based_filtering(
    outputdict: Dict, topk_confidence_ratio: int, cr_threshold: float
) -> Dict:
    outputdict_confidence_ratios = {}
    for source_iri, target_cands in outputdict.items():
        top_k_items = sorted(
            target_cands, key=lambda X: X[3] >= cr_threshold, reverse=True
        )[:topk_confidence_ratio]
        outputdict_confidence_ratios[source_iri] = top_k_items
    return outputdict_confidence_ratios


def confidence_score_based_filtering(
    outputdict_confidence_ratios: Dict,
    topk_confidence_score: int,
    llm_confidence_threshold: float,
    ir_score_threshold: float,
) -> List:
    filtered_predicts = []
    for source_iri, target_cands in outputdict_confidence_ratios.items():
        top_k_items = sorted(
            target_cands, key=lambda X: (X[2] >= llm_confidence_threshold), reverse=True
        )[:topk_confidence_score]
        for target, ir_score, llm_confidence, confidence_ratio in top_k_items:
            if ir_score >= ir_score_threshold:
                filtered_predicts.append(
                    {
                        "source": source_iri,
                        "target": target,
                        "score": ir_score,
                        "confidence": llm_confidence,
                        "confidence-ratio": confidence_ratio,
                    }
                )
    return filtered_predicts


def postprocess(
    predicts: List, topk_confidence_ratio: int = 3, topk_confidence_score: int = 1
) -> [List, Dict]:
    ir_outputs = predicts["generated-output"][0]["ir-outputs"]
    llm_outputs = predicts["generated-output"][1]["llm-output"]

    ir_outputs = preprocess_ir_outputs(predicts=ir_outputs)
    outputdict = build_outputdict(llm_outputs=llm_outputs, ir_outputs=ir_outputs)

    cr_threshold = threshold_finder(
        outputdict, index=3, use_lst=False
    )  # 3=confidence_ratio index
    outputdict_confidence_ratios = confidence_score_ratio_based_filtering(
        outputdict=outputdict,
        topk_confidence_ratio=topk_confidence_ratio,
        cr_threshold=cr_threshold,
    )

    ir_score_threshold = threshold_finder(
        outputdict_confidence_ratios, index=1, use_lst=True
    )  # 1=ir_score index
    llm_confidence_threshold = threshold_finder(
        outputdict, index=2, use_lst=False
    )  # 2=confidence_ratio index

    filtered_predicts = confidence_score_based_filtering(
        outputdict_confidence_ratios=outputdict_confidence_ratios,
        topk_confidence_score=topk_confidence_score,
        llm_confidence_threshold=llm_confidence_threshold,
        ir_score_threshold=ir_score_threshold,
    )
    configs = {
        "topk-confidence-ratio": topk_confidence_ratio,
        "topk-confidence-score": topk_confidence_score,
        "confidence-ratio-th": cr_threshold,
        "ir-score-th": ir_score_threshold,
        "llm-confidence-th": llm_confidence_threshold,
    }
    return filtered_predicts, configs
