# -*- coding: utf-8 -*-
from ontomap.base import BaseConfig
from ontomap.llm import LLMCatalog
from ontomap.ontology import ontology_matching
from ontomap.prompt import PromptCatalog
from ontomap.tools import workdir
from ontomap.utils import io


class NaivConfOAEIOMPipeline:
    def __init__(self, **kwargs) -> None:
        self.config = BaseConfig(approach=kwargs["approach"]).get_args(
            device=kwargs["device"]
        )
        self.load_from_json = kwargs["load-from-json"]
        self.approach = kwargs["approach"]
        if not kwargs["use-all-llm"]:
            self.llm_catalog = {}
            for llm_id, llm in LLMCatalog.items():
                if llm_id in kwargs["llms-to-consider"]:
                    self.llm_catalog[llm_id] = llm
        else:
            self.llm_catalog = LLMCatalog
        if not kwargs["use-all-approach-prompts"]:
            self.prompt_catalog = {}
            for prompt_type, prompt_module in PromptCatalog[kwargs["approach"]].items():
                if prompt_type in kwargs["approach-prompts-to-consider"]:
                    self.prompt_catalog[prompt_type] = prompt_module
        else:
            self.prompt_catalog = PromptCatalog[kwargs["approach"]]

    def __call__(self):
        for llm_id, llm in self.llm_catalog.items():
            LLM = llm(**vars(self.config)[llm_id])
            print(f"working on {llm_id}-{LLM}")
            for track, tasks in ontology_matching.items():
                print(f"\tWorking on {track} track")
                for task in tasks:
                    task_obj = task()
                    print(f"\tWorking on {task_obj} task")
                    if self.load_from_json:
                        task_owl = task_obj.load_from_json(
                            root_dir=self.config.root_dir
                        )
                    else:
                        task_owl = task_obj.collect(root_dir=self.config.root_dir)
                    for prompt_id, prompting in self.prompt_catalog.items():
                        print(f"\t\tPrompting ID is: {prompt_id}")
                        prompt = prompting()(**task_owl)
                        output_dict_obj = {
                            "llm": llm_id,
                            "llm-path": llm.path,
                            "llm-config": vars(self.config)[llm_id],
                            "dataset-info": task_owl["dataset-info"],
                            "prompt_id": prompt_id,
                            "prompt_template": prompting().get_prefilled_prompt(),
                        }
                        print("\t\tWorking on generating response!")
                        try:
                            llm_output = LLM.generate(input_data=prompt)
                        except RuntimeError as e:
                            print(f"MEMORY EXCEPTION: {e}")
                            llm_output = [str(e)]
                        output_dict_obj["generated_output"] = llm_output

                        print("\t\tCreate path to store data!")
                        # creating track_task_output_path file json path
                        try:
                            truncation = f"truncation={str(vars(self.config)[llm_id]['truncation'])}"
                        except Exception as e:
                            print(f"No truncation use here:{e}")
                            truncation = ""
                        track_task_output_path = workdir.make_output_dir(
                            output_dir=self.config.output_dir,
                            llm_id=llm_id,
                            dataset_info=task_owl["dataset-info"],
                            prompt_id=prompt_id,
                            approach=self.approach,
                            truncation=truncation,
                        )
                        print(f"\t\tStoring results in {track_task_output_path}.")
                        io.write_json(
                            output_path=track_task_output_path,
                            json_data=output_dict_obj,
                        )
                        print("\t\t" + "-" * 50)
                    print("\t" + "+" * 50)
