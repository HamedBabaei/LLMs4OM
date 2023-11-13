# -*- coding: utf-8 -*-
from ontomap.prompt.naiv_conv_oaei import (
    IRILabelChildrensInNaivePrompting,
    IRILabelDescInNaivePrompting,
    IRILabelInNaivePrompting,
    IRILabelParentsInNaivePrompting,
)

PromptCatalog = {
    "naiv-conv-oaei": {
        "iri-label": IRILabelInNaivePrompting,
        "iri-label-description": IRILabelDescInNaivePrompting,
        "iri-label-children": IRILabelChildrensInNaivePrompting,
        "iri-label-parent": IRILabelParentsInNaivePrompting,
    },
}


__all__ = ["PromptCatalog"]
