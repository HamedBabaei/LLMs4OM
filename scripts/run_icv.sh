#!/bin/bash

approach="icv"
encoder="rag"
use_all_encoders=False
approach_encoders_to_consider="['label']"
use_all_models=False
models_to_consider="['LLaMA7BAdaICV']"
load_from_json=True
device="cuda"
do_evaluation=False
outputs='outputs-icv'
batch_size=32

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
    'do-evaluation': $do_evaluation,
    'outputs-dir': '$outputs',
    'batch-size': '$batch_size'
}
print(f'Arguments are: {args}')
runner = OMPipelines(**args)
runner()
"
