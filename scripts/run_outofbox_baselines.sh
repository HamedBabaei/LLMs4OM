#!/bin/bash

approach="out-of-box"
use_all_approach_prompts=False
approach_prompts_to_consider=('iri-label' 'iri-label-description' 'iri-label-children' 'iri-label-parent')
use_all_llm=False
llms_to_consider=("FlanT5" "LLaMA7B" "LLaMA13B" "Wizard13B" "Mistral7B" "ChatGPT" "GPT4")
load_from_json=True
device="cpu"

# A function to join array elements with commas
join_by_comma() {
    local IFS=","
    echo "$*"
}

approach_prompts_to_consider_string="$(join_by_comma "${approach_prompts_to_consider[@]}")"
llms_to_consider_string="$(join_by_comma "${llms_to_consider[@]}")"

cd ..

python -c "
from ontomap import OntoMapPipeline
args = {
    'approach': '$approach',
    'use-all-approach-prompts': $use_all_approach_prompts,
    'approach-prompts-to-consider': ['$approach_prompts_to_consider_string'],
    'use-all-llm': $use_all_llm,
    'llms-to-consider': ['$llms_to_consider_string'],
    'load-from-json': $load_from_json,
    'device': '$device',
}
print(f'Arguments are: {args}')
runner = OntoMapPipeline(**args)
runner()
"
