# -*- coding: utf-8 -*-
from ontomap.encoder.lightweight import (
    IRILabelChildrensInLightweightEncoder,
    IRILabelDescInLightweightEncoder,
    IRILabelInLightweightEncoder,
    IRILabelParentsInLightweightEncoder,
)
from ontomap.encoder.naiv_conv_oaei import (
    IRILabelChildrensInNaiveEncoder,
    IRILabelDescInNaiveEncoder,
    IRILabelInNaiveEncoder,
    IRILabelParentsInNaiveEncoder,
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
}


__all__ = ["EncoderCatalog"]
