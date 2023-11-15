# -*- coding: utf-8 -*-

__version__ = "0.1.0"

import logging

from ontomap import base, encoder, evaluation, ontology, ontology_matchers, utils
from ontomap.pipeline import OMPipelines

__all__ = [
    "base",
    "ontology",
    "utils",
    "ontology_matchers",
    "encoder",
    "pipeline",
    "OMPipelines",
    "evaluation",
]

# Root logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stdout = logging.StreamHandler()
stdout.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
stdout.setFormatter(formatter)

logger.addHandler(stdout)
