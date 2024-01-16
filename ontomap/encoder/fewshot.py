# -*- coding: utf-8 -*-
from ontomap.encoder.rag import (
    IRILabelInRAGEncoder,
    IRILabelParentsInRAGEncoder,
    IRILabelChildrensInRAGEncoder
)


class IRILabelInFewShotEncoder(IRILabelInRAGEncoder):
    llm_encoder: str = "LabelFewShotDataset"


class IRILabelChildrensInFewShotEncoder(IRILabelChildrensInRAGEncoder):
    llm_encoder: str = "LabelChildrenFewShotDataset"


class IRILabelParentsInFewShotEncoder(IRILabelParentsInRAGEncoder):
    llm_encoder: str = "LabelParentFewShotDataset"
