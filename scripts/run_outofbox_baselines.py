# -*- coding: utf-8 -*-
# Imports
import __init__  # flake8-skip

from ontomap import pipeline

args = {
    "root_dataset_dir": "..",
    "approach": "out-of-box",
    "use-all-approach-prompts": True,
    "approach-prompts-to-consider": [
        "iri-label",
        "iri-label-description",
        "iri-label-children",
        "iri-label-parent",
    ],
    "use-all-llm": True,
    "llms-to-consider": [
        "FlanT5",
        "LLaMA7B",
        "LLaMA13B",
        "Wizard13B",
        "Mistral7B",
        "ChatGPT",
        "GPT4",
    ],
    "load-from-json": True,
    "device": "cpu",
}

runner = pipeline.OntoMapPipeline(**args)
runner()
