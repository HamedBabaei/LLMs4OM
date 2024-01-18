# -*- coding: utf-8 -*-
from ontomap.ontology_matchers.rag.rag import RAG
from ontomap.ontology_matchers.fewshot.dataset import * # NOQA
from ontomap.postprocess import process
from typing import List, Any

import random


class FewShot(RAG):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_shot = self.kwargs['nshots']

    def __str__(self):
        return "FewShotRAG"

    def generate(self, input_data: List) -> List:
        """
        :param input_data:
                {
                    "retriever-encoder": self.retrieval_encoder,
                    "task-args": kwargs,
                    "source-onto-iri2index": source_onto_iri2index,
                    "target-onto-iri2index": target_onto_iri2index
                }
        :return:
        """
        # IR generation
        ir_output = self.ir_generate(input_data=input_data)
        ir_output_cleaned = process.preprocess_ir_outputs(predicts=ir_output)
        examples = self.build_fewshots(input_data=input_data)
        input_data['examples'] = examples
        # LLm generation
        llm_predictions = self.llm_generate(
            input_data=input_data, ir_output=ir_output_cleaned
        )
        return [{"ir-outputs": ir_output}, {"llm-output": llm_predictions}, {"fewshot-samples": examples}]

    def build_llm_encoder(self, input_data: Any, llm_inputs: Any) -> Any:
        dataset = eval(input_data["llm-encoder"])(data=llm_inputs, few_shot_examples=True)
        dataset.build_exemplars(examples=input_data['examples'])
        return dataset

    def build_fewshots(self, input_data: List) -> List:

        def minor_clean(concept: str) -> str:
            concept = concept.replace("_", " ")
            concept = concept.lower()
            return concept

        track = input_data['task-args']['dataset-info']['track']
        if track == 'bio-ml':
            reference = input_data['task-args']['reference']['equiv']['train']
        else:
            reference = input_data['task-args']['reference']

        random_positive_examples = []
        for ref in reference:
            try:
                source_iri, target_iri = ref['source'], ref['target']
                source = input_data['task-args']['source'][input_data['source-onto-iri2index'][source_iri]]['label']
                target = input_data['task-args']['target'][input_data['target-onto-iri2index'][target_iri]]['label']
                if minor_clean(source) != minor_clean(target):
                    random_positive_examples.append([minor_clean(source), minor_clean(target)])
            except Exception as err:
                print(f"ERROR OCCURED! {err}")
            if len(random_positive_examples) == self.n_shots:
                break

        random_negative_examples = []
        for ref in reference:
            source_iri, target_iri = ref['source'], ref['target']
            source = input_data['task-args']['source'][input_data['source-onto-iri2index'][source_iri]]['label']
            target = input_data['task-args']['target'][input_data['target-onto-iri2index'][target_iri]]['label']
            for neg_ref in reference:
                try:
                    neg_source_iri, neg_target_iri = neg_ref['source'], neg_ref['target']
                    neg_source = input_data['task-args']['source'][input_data['source-onto-iri2index'][neg_source_iri]]['label']
                    neg_target = input_data['task-args']['target'][input_data['target-onto-iri2index'][neg_target_iri]]['label']
                    if minor_clean(neg_source) != minor_clean(source) and minor_clean(target) != minor_clean(
                            neg_target) and minor_clean(neg_source) != minor_clean(neg_target):
                        random_negative_examples.append([minor_clean(source), minor_clean(neg_target)])
                        break
                except Exception as err:
                    print(f"ERROR OCCURED! {err}")
            if len(random_negative_examples) == self.n_shots:
                break

        fewshot_examples = [{'source': source, 'target': target, 'label': 'yes'} for source, target in random_positive_examples] + \
                           [{'source': source, 'target': target, 'label': 'no'} for source, target in random_negative_examples]
        random.shuffle(fewshot_examples)
        return fewshot_examples
