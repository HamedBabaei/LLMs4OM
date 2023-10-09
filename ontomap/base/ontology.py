# -*- coding: utf-8 -*-
import os
from abc import ABC
from typing import Any, Dict, List

from owlready2 import World
from rdflib import Namespace, URIRef


def load_ontology(input_file_path: str) -> World:
    ontology = World()
    ontology.get_ontology(input_file_path).load()
    return ontology


class BaseOntologyParser(ABC):
    def is_contain_label(self, owl_class: Any) -> bool:
        if len(owl_class.label) == 0:
            return False
        return True

    def get_name(self, owl_class: Any) -> str:
        return owl_class.name

    def get_label(self, owl_class: Any) -> str:
        return owl_class.label.first()

    def get_iri(self, owl_class: Any) -> str:
        return owl_class.iri

    def get_subclasses(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.subclasses())

    def get_ancestors(self, owl_class: Any) -> List:
        ans = self.get_owl_items(owl_class.ancestors())
        return ans

    def get_synonyms(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.hasRelatedSynonym)

    def get_comments(self, owl_class: Any) -> List:
        pass

    def get_owl_items(self, owl_class: Any) -> List:
        owl_items = []
        for item in owl_class:
            if self.is_contain_label(item):
                owl_items.append(
                    {
                        "iri": self.get_iri(item),
                        "name": self.get_name(item),
                        "label": self.get_label(item),
                    }
                )
        return owl_items

    def extract_data(self, ontology: Any) -> List[Dict]:
        parsed_ontology = []
        for owl_class in ontology.classes():
            if not self.is_contain_label(owl_class):
                continue
            owl_class_info = {
                "name": self.get_name(owl_class),
                "iri": self.get_iri(owl_class),
                "label": self.get_label(owl_class),
                "subclasses": self.get_subclasses(owl_class),
                "ancestors": self.get_ancestors(owl_class),
                "synonyms": self.get_synonyms(owl_class),
                "comment": self.get_comments(owl_class),
            }
            parsed_ontology.append(owl_class_info)
        return parsed_ontology

    def parse(self, root_dir: str, ontology_file_name: str) -> List:
        input_file_path = os.path.join(root_dir, ontology_file_name)
        ontology = load_ontology(input_file_path=input_file_path)
        return self.extract_data(ontology)


class BaseAlignmentsParser(ABC):
    namespace: Namespace = Namespace(
        "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
    )
    entity_1: URIRef = URIRef(namespace + "entity1")
    entity_2: URIRef = URIRef(namespace + "entity2")
    relation: URIRef = URIRef(namespace + "relation")

    def extract_data(self, reference: Any) -> List[Dict]:
        parsed_references = []
        graph = reference.as_rdflib_graph()
        for source, predicate, target in graph:
            if predicate == self.relation:
                entity_1 = [
                    str(o) for s, p, o in graph.triples((source, self.entity_1, None))
                ][0]
                entity_2 = [
                    str(o) for s, p, o in graph.triples((source, self.entity_2, None))
                ][0]
                parsed_references.append(
                    {"source": entity_1, "target": entity_2, "relation": str(target)}
                )
        return parsed_references

    def parse(self, root_dir: str, reference_file_name: str) -> List:
        input_file_path = os.path.join(root_dir, reference_file_name)
        reference = load_ontology(input_file_path=input_file_path)
        return self.extract_data(reference)
