# -*- coding: utf-8 -*-
from typing import Any

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ontomap.ontology_matchers.retrieval.retrieval import BiEncoderRetrieval, Retrieval


class BERTRetrieval(BiEncoderRetrieval):
    path: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    def __str__(self):
        return super().__str__() + "+BERTRetrieval"


class SpecterBERTRetrieval(BiEncoderRetrieval):
    path: str = "allenai/specter_plus_plus"

    def __str__(self):
        return super().__str__() + "SpecterBERTRetrieval"


class FlanT5XLRetrieval(BiEncoderRetrieval):
    path: str = "google/flan-t5-xl"

    def __str__(self):
        return super().__str__() + "FlanT5XLRetrieval"


class FlanT5XXLRetrieval(BiEncoderRetrieval):
    path: str = "google/flan-t5-xxl"

    def __str__(self):
        return super().__str__() + "FlanT5XLRetrieval"


class TFIDFRetrieval(Retrieval):
    path: str = "NO MODEL LOADING IN TFIDFRetrieval MODEL"

    def load(self):
        self.model = TfidfVectorizer()

    def fit(self, inputs: Any) -> Any:
        self.model.fit(inputs)
        return self.transform(inputs=inputs)

    def transform(self, inputs: Any) -> Any:
        return self.model.transform(inputs)

    def estimate_similarity(self, query_embed: Any, candidate_embeds: Any) -> Any:
        return cosine_similarity(query_embed, candidate_embeds).reshape((-1,))

    def __str__(self):
        return super().__str__() + "+TFIDFRetrieval"


class BM25Retrieval(Retrieval):
    path: str = "NO MODEL LOADING IN TFIDFRetrieval MODEL"

    def fit(self, inputs: Any) -> Any:
        tokenized_inputs = [input.split(" ") for input in inputs]
        self.model = BM25Okapi(tokenized_inputs)
        return None

    def transform(self, inputs: Any) -> Any:
        return [input.split(" ") for input in inputs]

    def estimate_similarity(self, query_embed: Any, candidate_embeds: Any) -> Any:
        docs_scores = self.model.get_scores(query_embed)
        return docs_scores

    def __str__(self):
        return super().__str__() + "+BM25Retrieval"
