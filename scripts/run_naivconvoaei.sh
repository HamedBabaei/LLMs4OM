#!/bin/bash

approach="naiv-conv-oaei"
encoder="naiv-conv-oaei"
use_all_encoders=False
approach_encoders_to_consider="['iri-label', 'iri-label-description', 'iri-label-children', 'iri-label-parent']"
use_all_models=False
models_to_consider="['FlanT5','LLaMA7B','LLaMA13B','Wizard13B','Mistral7B', 'ChatGPT', 'GPT4']"
load_from_json=True
device="cpu"
do_evaluation=False

cd ..

python -c "
from ontomap import OMPipelines
args = {
    'approach': '$approach',
    'encoder': '$encoder',
    'use-all-encoders': $use_all_encoders,
    'approach-encoders-to-consider': $approach_encoders_to_consider,
    'use-all-models': $use_all_models,
    'models-to-consider': $models_to_consider,
    'load-from-json': $load_from_json,
    'device': '$device',
    'do-evaluation': $do_evaluation
}
print(f'Arguments are: {args}')
runner = OMPipelines(**args)
runner()
"
