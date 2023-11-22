# -*- coding: utf-8 -*-
from ontomap.pipeline.oaei_pipeline import OAEIOMPipeline

OMPipelines = {
    "naiv-conv-oaei": OAEIOMPipeline,
    "lightweight": OAEIOMPipeline,
    "retrieval": OAEIOMPipeline,
    "rag": OAEIOMPipeline,
}

__all__ = ["OMPipelines"]
