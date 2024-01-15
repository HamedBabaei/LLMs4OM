# -*- coding: utf-8 -*-
from typing import Any

from ontomap.encoder.encoders import RAGEncoder
from ontomap.encoder.lightweight import (
    IRILabelInLightweightEncoder,
)


class IRILabelInRAGEncoder(RAGEncoder):
    items_in_owl: str = "(Label)"
    retrieval_encoder: Any = IRILabelInLightweightEncoder
    llm_encoder: str = "LabelRAGDataset"


class IRILabelChildrensInRAGEncoder(RAGEncoder):
    items_in_owl: str = "(Label, Children)"
    retrieval_encoder: Any = IRILabelInLightweightEncoder
    llm_encoder: str = "LabelChildrenRAGDataset"


class IRILabelParentsInRAGEncoder(RAGEncoder):
    items_in_owl: str = "(Label, Parent)"
    retrieval_encoder: Any = IRILabelInLightweightEncoder
    llm_encoder: str = "LabelParentRAGDataset"
