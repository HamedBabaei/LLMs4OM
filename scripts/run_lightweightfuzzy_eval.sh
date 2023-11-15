#!/bin/bash

approach="lightweight"
use_all_encoders=True
approach_encoders_to_consider="['label', 'label-description', 'label-children', 'label-parent']"
use_all_models=False
models_to_consider="['FuzzySM']"
load_from_json=True
device="cpu"
do_evaluation=True

cd ..

python -c "
from ontomap import OMPipelines
args = {
    'approach': '$approach',
    'use-all-encoders': $use_all_encoders,
    'approach-encoders-to-consider': $approach_encoders_to_consider,
    'use-all-models': $use_all_models,
    'models-to-consider': $models_to_consider,
    'load-from-json': $load_from_json,
    'device': '$device',
    'do-evaluation': $do_evaluation
}
print(f'Arguments are: {args}')
runner = OMPipelines['$approach'](**args)
runner()
"
