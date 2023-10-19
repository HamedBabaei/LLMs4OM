# -*- coding: utf-8 -*-
from ontomap.prompt.baselines_prompt import (
    IRILabelAncInBaselinePrompting,
    IRILabelDescInBaselinePrompting,
    IRILabelInBaselinePrompting,
    IRILabelSubClssInBaselinePrompting,
)

PromptCatalog = {
    "baseline": {
        "iri-label": IRILabelInBaselinePrompting,
        "iri-label-desc": IRILabelDescInBaselinePrompting,
        "iri-label-subclss": IRILabelSubClssInBaselinePrompting,
        "iri-label-anc": IRILabelAncInBaselinePrompting,
    },
}


__all__ = ["PromptCatalog"]
