# -*- coding: utf-8 -*-
from typing import Any, List

import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ontomap.base import BaseOMModel


class Retrieval(BaseOMModel):
    path: str = ""
    model: Any = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load()

    def load(self):
        pass

    def __str__(self):
        return "Retrieval"

    def fit(self, inputs: Any) -> Any:
        pass

    def transform(self, inputs: Any) -> Any:
        pass

    def estimate_similarity(self, query_embed: Any, candidate_embeds: Any) -> Any:
        pass

    def get_top_k(self, query_embed: Any, candidate_embeds: Any) -> [List, List]:
        results = self.estimate_similarity(
            query_embed=query_embed, candidate_embeds=candidate_embeds
        )
        values = [(score, index) for index, score in enumerate(results)]
        dtype = [("score", float), ("index", int)]
        results = np.array(values, dtype=dtype)
        top_k_items = np.sort(results, order="score")[-self.kwargs["top_k"] :][::-1]
        top_k_indexes, top_k_scores = [], []
        for top_k in top_k_items:
            top_k_scores.append(top_k[0])
            top_k_indexes.append(top_k[1])
        return top_k_indexes, top_k_scores

    def generate(self, input_data: List) -> List:
        source_ontology = input_data[0]
        target_ontology = input_data[1]
        predictions = []

        candidates_embedding = self.fit(
            inputs=[target["text"] for target in target_ontology]
        )
        queries_embedding = self.transform(
            inputs=[source["text"] for source in source_ontology]
        )

        for source_id, query_embed in tqdm(enumerate(queries_embedding)):
            ids, scores = self.get_top_k(
                query_embed=query_embed, candidate_embeds=candidates_embedding
            )
            candidates_iris, candidates_scores = [], []
            for candidate_id, candidate_score in zip(ids, scores):
                candidates_iris.append(target_ontology[candidate_id]["iri"])
                candidates_scores.append(candidate_score)
            if len(candidates_iris) != 0:
                predictions.append(
                    {
                        "source": source_ontology[source_id]["iri"],
                        "target-cands": candidates_iris,
                        "score-cands": candidates_scores,
                    }
                )
        return predictions


class BiEncoderRetrieval(Retrieval):
    path: str = ""

    def load(self):
        self.model = SentenceTransformer(self.path, device=self.kwargs["device"])

    def fit(self, inputs: Any) -> Any:
        return self.model.encode(inputs)

    def transform(self, inputs: Any) -> Any:
        return self.model.encode(inputs)

    def estimate_similarity(self, query_embed: Any, candidate_embeds: Any) -> Any:
        return np.dot(candidate_embeds, query_embed) / (
            norm(candidate_embeds, axis=1) * norm(query_embed)
        )
