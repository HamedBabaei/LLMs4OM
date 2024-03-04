<div align="center">
 <img src="images/logo.png"/>
</div>

<div align="center">


[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

The supplementary material for detailed results of retriever and LLM modules is available here for download: [![The supplementary material](https://img.shields.io/badge/Download%20AS%20pdf-EF3939?style=badge&logo=adobeacrobatreader&logoColor=white&color=black&labelColor=ec1c24)](docs/LLMs4OM_Supplementary_Material.pdf)

## What is the LLMs4OM?

The LLMs4OM framework is a novel approach for effective Ontology Matching (OM) using LLMs. This framework utilizes two modules for retrieval and matching, respectively, enhanced by zero-shot prompting across three ontology representations: concept, concept-parent, and concept-children.  It is capable of comprehensive evaluations using 20 OM datasets (but not limited to) from various domains. The LLMs4OM framework, can match and even surpass the performance of traditional OM systems, particularly in complex matching scenarios.

The following diagram represent the LLMs4OM framework.
<div align="center">
 <img src="images/LLMs4OM.jpg"/>
</div>

The LLMs4OM framework offers a retrieval augmented generation (RAG) approach within LLMs for OM. LLMs4OM uses $O_{source}$ as query $Q(O_{source})$ to retrieve possible matches for for any $C_s \in C_{source}$ from $C_{target} \in O_{target}$. Where, $C_{target}$ is stored in the knowledge base $KB(O_{target})$. Later, $C_{s}$ and obtained $C_t \in C_{target}$ are used to query the LLM to check whether the $(C_s, C_t)$ pair is a match. As shown in above diagram, the framework comprises four main steps: 1) Concept representation, 2) Retriever model, 3) LLM, and 4) Post-processing.

## Installation

You can also install and use the LLMs4OM using the following commands.
```
git clone https://github.com/XXX/LLMs4OM.git
cd LLMs4OM

pip install -r requirements.txt
mv .env-example .env
```
Next, update your tokens in `.env` or if you don't want to use `LLaMA-2` or `GPT-3.5` LLMs just put dummy tokens there.
Once you installed the requirements and prepared the `.env` file, you can move forward with experimentation.

## Quick Tour
```python
from ontomap.ontology import MouseHumanOMDataset
from ontomap.base import BaseConfig
from ontomap.evaluation.evaluator import evaluator
from ontomap.encoder import IRILabelInRAGEncoder
from ontomap.ontology_matchers import MistralLLMBertRAG
from ontomap.postprocess import process

# Setting configurations for experimenting 'rag' on GPU with batch size of 16
config = BaseConfig(approach='rag').get_args(device='cuda', batch_size=16)
# set dataset directory
config.root_dir = "datasets"
# parse task source, target, and reference ontology
ontology = MouseHumanOMDataset().collect(root_dir=config.root_dir)

# init encoder (concept-representation)
encoded_inputs = IRILabelInRAGEncoder()(ontology)

# init Mistral-7B + BERT
model = MistralLLMBertRAG(config.MistralBertRAG)
# generate results
predicts = model.generate(input_data=encoded_inputs)

# post-processing
predicts, _ = process.postprocess_hybrid(predicts=predicts,
                                         llm_confidence_th=0.7,
                                         ir_score_threshold=0.9)
# evaluation
results = evaluator(track='anatomy',
                    predicts=predicts,
                    references=ontology["reference"])
print(results)
```

## Citation
If you found this project useful in your work or research please cite it by using this BibTeX entry:
```bibtex
@misc{llms4om2024,
  title =        {LLMs4OM: Matching Ontologies with Large Language Models},
  year =         {2024}
}
```
