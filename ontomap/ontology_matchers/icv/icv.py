# -*- coding: utf-8 -*-

from typing import List

import torch
from torch.nn import functional as F

from ontomap.ontology_matchers.icv.tasks.demo import DemoProbInferenceForStyle
from ontomap.ontology_matchers.rag.rag import RAG, RAGBasedDecoderLLMArch
from ontomap.postprocess import process


def tokenize_each_demonstration(tok, demonstration_list, dataset_name=None):
    tokenized_demonstration_list = []
    for exp_id in range(len(demonstration_list)):
        demonstration_list[exp_id] = (
            demonstration_list[exp_id][0].strip(" .").strip("."),
            demonstration_list[exp_id][1].strip(" .").strip("."),
        )
        print(demonstration_list)
        e_original = tok(demonstration_list[exp_id][0])
        e_rewrite = tok(demonstration_list[exp_id][1])
        tokenized_demonstration_list.append((e_original, e_rewrite))
    return tokenized_demonstration_list


class AdapterLayer(torch.nn.Module):
    def __init__(self, icvs, alpha):
        super(AdapterLayer, self).__init__()
        self.icvs = icvs
        self.alpha = alpha
        self.weight_all = []

    def forward(self, x):
        input_dtype = x.dtype
        if self.icvs is not None:
            norm = torch.norm(x.float(), dim=-1).unsqueeze(-1)
            alpha = self.alpha
            icv_all_tasks = 0
            for i in range(len(self.icvs)):
                lambda_sim = 1.0 + torch.max(
                    torch.tensor([0.0]).to(x.device),
                    F.cosine_similarity(x.float(), self.icvs[i][None, None, :], dim=-1),
                ).unsqueeze(-1)
                icv_all_tasks -= (
                    alpha[i]
                    * lambda_sim
                    * F.normalize(self.icvs[i], dim=-1).repeat(1, x.shape[1], 1)
                )
            icv_all_tasks = 0.1 * icv_all_tasks / len(self.icvs)

            x = (
                F.normalize(F.normalize(x.float(), dim=-1) + icv_all_tasks, dim=-1)
                * norm
            )
            return x.type(input_dtype)
        else:
            return x


class ICVAdapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, icvs, alpha):
        for i in range(0, len(self.model.transformer.h)):
            icvs_ = icvs[i]
            self.model.transformer.h[i].mlp = torch.nn.Sequential(
                self.model.transformer.h[i].mlp, AdapterLayer(icvs_, alpha)
            )
        return self.model

    def remove_adapter(self):
        weight_all = []

        for i in range(0, len(self.model.transformer.h)):
            weight_all.append(self.model.transformer.h[i].mlp[1].weight_all)
            self.model.transformer.h[i].mlp = self.model.transformer.h[i].mlp[0]
        return weight_all


class ICVBasedDecoderLLMArch(RAGBasedDecoderLLMArch):
    icv_dataset = "demo"
    icv_prompt_version = "default"
    icv_kv_iter = 15
    icv_step_size = 0.01
    icv_num_k_shots = 1
    icv_momentum = 0.9
    icv_alpha = 1.0
    icv_seed = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_agent = DemoProbInferenceForStyle(
            prompt_version=self.icv_prompt_version
        )
        self.task_agent.set_seed(self.icv_seed)

    def __str__(self):
        return "ICVBasedDecoderLLMArch"

    def build_icv(self, examples):
        icv_examples = self.task_agent.get_icv(
            self.model, tokenize_each_demonstration(self.tokenizer, examples)
        )
        icvs_to_shift = [icv_examples]
        updated_wrapper = ICVAdapter(self.model)
        _ = updated_wrapper.get_model(
            torch.stack(icvs_to_shift, dim=1).cuda(), alpha=[self.icv_alpha]
        )


class ICV(RAG):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self):
        return "ICV"

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
        ir_output = self._ir_generate(input_data=input_data)
        ir_output_cleaned = process.preprocess_ir_outputs(predicts=ir_output)
        examples = self.build_icv_examples()
        self.LLM.build_icv(examples=examples)
        # LLm generation
        llm_predictions = self._llm_generate(
            input_data=input_data, ir_output=ir_output_cleaned
        )
        return [{"ir-outputs": ir_output}, {"llm-output": llm_predictions}]

    def build_icv_examples(self):
        return []
