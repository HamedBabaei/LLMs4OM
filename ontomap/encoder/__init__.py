# -*- coding: utf-8 -*-
from ontomap.encoder.lightweight import (
    IRILabelChildrensInLightweightEncoder,
    IRILabelDescInLightweightEncoder,
    IRILabelInLightweightEncoder,
    IRILabelParentsInLightweightEncoder,
)
from ontomap.encoder.naivconvoaei import (
    IRILabelChildrensInNaiveEncoder,
    IRILabelDescInNaiveEncoder,
    IRILabelInNaiveEncoder,
    IRILabelParentsInNaiveEncoder,
)
from ontomap.encoder.rag import (
    IRILabelChildrensInRAGEncoder,
    IRILabelInRAGEncoder,
    IRILabelParentsInRAGEncoder,
)
from ontomap.encoder.fewshot import (
    IRILabelParentsInFewShotEncoder,
    IRILabelInFewShotEncoder,
    IRILabelChildrensInFewShotEncoder
)


EncoderCatalog = {
    "naiv-conv-oaei": {
        "iri-label": IRILabelInNaiveEncoder,
        "iri-label-description": IRILabelDescInNaiveEncoder,
        "iri-label-children": IRILabelChildrensInNaiveEncoder,
        "iri-label-parent": IRILabelParentsInNaiveEncoder,
    },
    "lightweight": {
        "label": IRILabelInLightweightEncoder,
        "label-description": IRILabelDescInLightweightEncoder,
        "label-children": IRILabelChildrensInLightweightEncoder,
        "label-parent": IRILabelParentsInLightweightEncoder,
    },
    "rag": {
        "label": IRILabelInRAGEncoder,
        "label-children": IRILabelChildrensInRAGEncoder,
        "label-parent": IRILabelParentsInRAGEncoder,
    },
    "fewshot": {
        "label": IRILabelInFewShotEncoder,
        "label-children": IRILabelChildrensInFewShotEncoder,
        "label-parent": IRILabelParentsInFewShotEncoder

    }
}


__all__ = ["EncoderCatalog"]
