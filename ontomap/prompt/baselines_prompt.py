# -*- coding: utf-8 -*-
from typing import Dict

from ontomap.base import BasePrompt

PROMPT = """<Problem Definition>
In this task, we are given two ontologies in the form of {items_in_owl}, which consist of IRI and classes.

<Ontologies-1>
{source}

<Ontologies-2>
{target}

<Objective>
Our objective is to provide ontology mapping for the provided ontologies based on their semantic similarities.

For a class in the first ontology, which class in ontology 2 is the best match?

List matches as a List.
"""


class BaselinePrompting(BasePrompt):
    prompt_template: str = PROMPT
    items_in_owl: str = ""

    def parse(self, **kwargs) -> str:
        source_onto, target_onto = kwargs["source"], kwargs["target"]
        source_text = ""
        for source in source_onto:
            source_text += self.get_owl_items(owl=source)
        target_text = ""
        for target in target_onto:
            target_text += self.get_owl_items(owl=target)
        prompt_sample = self.get_prefilled_prompt()
        prompt_sample = prompt_sample.replace("{source}", source_text)
        prompt_sample = prompt_sample.replace("{target}", target_text)
        return prompt_sample

    def __str__(self):
        return {"Template": super().__str__(), "BaselinePrompting": self.items_in_owl}

    def get_owl_items(self, owl: Dict) -> str:
        pass

    def get_prefilled_prompt(self) -> str:
        prompt_sample = self.prompt_template
        prompt_sample = prompt_sample.replace("{items_in_owl}", self.items_in_owl)
        return prompt_sample


class IRILabelInBaselinePrompting(BaselinePrompting):
    items_in_prompt: str = "(IRI, Label)"

    def get_owl_items(self, owl: Dict) -> str:
        return f"({owl['iri']}, {owl['label']}), "


class IRILabelDescInBaselinePrompting(BaselinePrompting):
    items_in_prompt: str = "(IRI, Label, Description)"

    def get_owl_items(self, owl: Dict) -> str:
        return f"({owl['iri']}, {owl['label']}, {str(owl['comment'])}), "


class IRILabelSubClssInBaselinePrompting(BaselinePrompting):
    items_in_prompt: str = "(IRI, Label, SubClasses)"

    def get_owl_items(self, owl: Dict) -> str:
        sub_classes = [subclasses["label"] for subclasses in owl["subclasses"]]
        return f"({owl['iri']}, {owl['label']}, {str(sub_classes)}), "


class IRILabelAncInBaselinePrompting(BaselinePrompting):
    items_in_prompt: str = "(IRI, Label, Ancestors)"

    def get_owl_items(self, owl: Dict) -> str:
        ancestors = [anc["label"] for anc in owl["ancestors"]]
        return f"({owl['iri']}, {owl['label']}, {str(ancestors)}), "
