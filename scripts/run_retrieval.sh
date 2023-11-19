#!/bin/bash

approach="retrieval"
encoder="lightweight"
use_all_encoders=True
approach_encoders_to_consider="['label', 'label-description', 'label-children', 'label-parent']"
use_all_models=False
models_to_consider="['BM25Retrieval', 'TFIDFRetrieval', 'BERTRetrieval', 'SpecterBERTRetrieval', 'FlanT5XLRetrieval', 'FlanT5XXRetrieval']"
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
print('start2')
runner = OMPipelines['$approach'](**args)
runner()
"
