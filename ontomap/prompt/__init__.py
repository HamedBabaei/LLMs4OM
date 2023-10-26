# -*- coding: utf-8 -*-
from ontomap.prompt.out_of_box import (
    IRILabelChildrensInOutOfBoxPrompting,
    IRILabelDescInOutOfBoxPrompting,
    IRILabelInOutOfBoxPrompting,
    IRILabelParentsInOutOfBoxPrompting,
)

PromptCatalog = {
    "out-of-box": {
        "iri-label": IRILabelInOutOfBoxPrompting,
        "iri-label-description": IRILabelDescInOutOfBoxPrompting,
        "iri-label-children": IRILabelChildrensInOutOfBoxPrompting,
        "iri-label-parent": IRILabelParentsInOutOfBoxPrompting,
    },
}


__all__ = ["PromptCatalog"]
