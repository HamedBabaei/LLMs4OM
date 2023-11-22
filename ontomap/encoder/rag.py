# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from ontomap.encoder.encoders import RAGEncoder
from ontomap.encoder.lightweight import IRILabelInLightweightEncoder


class IRILabelInRAGEncoder(RAGEncoder):
    items_in_owl: str = "(Label)"
    retrieval_encoder: Any = IRILabelInLightweightEncoder
    llm_encoder: Any = None
