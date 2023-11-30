# -*- coding: utf-8 -*-
from typing import Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ontomap.base import BaseOMModel
from ontomap.ontology_matchers.llm.llm import LLaMA2DecoderLLMArch, OpenAILLMArch
from ontomap.ontology_matchers.rag.dataset import *
from ontomap.postprocess.filtering import refactor_retrieval_predicts


class RAGBasedDecoderLLMArch(LLaMA2DecoderLLMArch):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.index2label = {0: "yes", 1: "no"}
        self.label2index = [
            self.tokenizer("yes").input_ids[-1],
            self.tokenizer("no").input_ids[-1],
        ]

    def __str__(self):
        return "RAGBasedDecoderLLMArch"

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized_input_data,
                pad_token_id=self.tokenizer.eos_token_id,
                # num_beams=self.kwargs["num_beams"],
                max_new_tokens=self.kwargs["max_token_length"],
                # temperature=self.kwargs["temperature"],
                # top_p=self.kwargs["top_p"],
                output_scores=True,
                return_dict_in_generate=True
            )
        probas = outputs.scores[0][:, self.label2index].softmax(-1)
        probas_per_candidate_tokens = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_candidate_tokens.values]
        sequences = [
            self.index2label[int(indice)]
            for indice in probas_per_candidate_tokens.indices
        ]
        return [sequences, sequence_probas]

    def generate_for_multiple_input(self, tokenized_input_data: Any) -> List:
        return self.generate_for_one_input(tokenized_input_data=tokenized_input_data)


class RAGBasedOpenAILLMArch(OpenAILLMArch):
    def __str__(self):
        return "RAGBasedOpenAILLMArch"

    def post_processor(self, generated_texts: List) -> List:
        sequences, sequence_probas = [], []
        for generated_text in generated_texts:
            processed_output = generated_text["choices"][0]["message"][
                "content"
            ].lower()
            proba = 1
            if "yes" in processed_output:
                processed_output = "yes"
            else:
                processed_output = "no"
            sequences.append(processed_output)
            sequence_probas.append(proba)
        return [sequences, sequence_probas]


class RAG(BaseOMModel):
    path: str = "NO MODEL LOADING IN RAG MODELS"
    Retrieval = None
    LLM = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Retrieval = self.Retrieval(**self.kwargs["retriever-config"])
        self.LLM = self.LLM(**self.kwargs["llm-config"])

    def __str__(self):
        return "RAG"

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
        ir_output = self._ir_generate(args=input_data)
        # Refactor IR outputs
        ir_output_cleaned = refactor_retrieval_predicts(predicts=ir_output)
        source_onto_iri2index, target_onto_iri2index = (
            input_data["source-onto-iri2index"],
            input_data["target-onto-iri2index"],
        )
        source_onto, target_onto = (
            input_data["task-args"]["source"],
            input_data["task-args"]["target"],
        )

        # Build LLm inputs
        llm_inputs = []
        for retrieved_items in ir_output_cleaned:
            llm_inputs.append(
                {
                    "source": source_onto[
                        source_onto_iri2index[retrieved_items["source"]]
                    ],
                    "target": target_onto[
                        target_onto_iri2index[retrieved_items["target"]]
                    ],
                    "ir-scores": retrieved_items["score"],
                }
            )
        # print(llm_inputs[0])
        # create DataLoader for batching!
        dataset = eval(input_data["llm-encoder"])(
            data=llm_inputs, llm_id=input_data["task-args"]["llm"]
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.kwargs["llm-config"]["batch_size"],
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        # Inference model!
        predictions = []
        for batch in tqdm(dataloader):
            texts, iris = batch["texts"], batch["iris"]
            sequences, sequence_probas = self.LLM.generate(texts)
            for label, proba, iri_pair in zip(sequences, sequence_probas, iris):
                if label == "yes":
                    predictions.append(
                        {"source": iri_pair[0], "target": iri_pair[1], "score": proba}
                    )
        return [{"ir-outputs": ir_output}, {"llm-output": predictions}]

    def _llm_generate(self, args: Any, ir_output: Any) -> List:
        pass

    def _ir_generate(self, args: Any) -> Any:
        """
        :param args:
                {
                    "retriever-encoder": self.retrieval_encoder,
                    "llm-encoder": self.llm_encoder,
                    "task-args": kwargs,
                    "source-onto-iri2index": source_onto_iri2index,
                    "target-onto-iri2index": target_onto_iri2index
                }
        :return:
        """
        retrieval_input = args["retriever-encoder"]()(**args["task-args"])
        retrieval_predicts = self.Retrieval.generate(input_data=retrieval_input)
        return retrieval_predicts
