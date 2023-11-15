# -*- coding: utf-8 -*-
import os

from ontomap.base import BaseConfig
from ontomap.encoder import EncoderCatalog
from ontomap.evaluation import evaluator_module
from ontomap.ontology import ontology_matching
from ontomap.ontology_matchers import MatcherCatalog
from ontomap.tools import workdir
from ontomap.utils import io


class OAEIOMPipeline:
    def __init__(self, **kwargs) -> None:
        self.do_evaluation = kwargs["do-evaluation"]
        self.config = BaseConfig(approach=kwargs["approach"]).get_args(
            device=kwargs["device"]
        )
        self.load_from_json = kwargs["load-from-json"]
        self.approach = kwargs["approach"]
        if not kwargs["use-all-models"]:
            self.matcher_catalog = {}
            for model_id, model in MatcherCatalog[kwargs["approach"]].items():
                if model_id in kwargs["models-to-consider"]:
                    self.matcher_catalog[model_id] = model
        else:
            self.matcher_catalog = MatcherCatalog
        if not kwargs["use-all-encoders"]:
            self.encoder_catalog = {}
            for encoder_type, encoder_module in EncoderCatalog[
                kwargs["approach"]
            ].items():
                if encoder_type in kwargs["approach-encoders-to-consider"]:
                    self.encoder_catalog[encoder_type] = encoder_module
        else:
            self.encoder_catalog = EncoderCatalog[kwargs["approach"]]

    def __call__(self):
        for model_id, matcher_model in self.matcher_catalog.items():
            if not self.do_evaluation:
                MODEL = matcher_model(**vars(self.config)[model_id])
                print(f"working on {model_id}-{MODEL}")
            else:
                print(f"Run evaluation on {model_id}")
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
                    for encoder_id, encoder_module in self.encoder_catalog.items():
                        print(f"\t\tPrompting ID is: {encoder_id}")
                        if not self.do_evaluation:
                            encoded_inputs = encoder_module()(**task_owl)
                            output_dict_obj = {
                                "model": model_id,
                                "model-path": matcher_model.path,
                                "model-config": vars(self.config)[model_id],
                                "dataset-info": task_owl["dataset-info"],
                                "encoder-id": encoder_id,
                                "encoder-info": encoder_module().get_encoder_info(),
                            }
                            print("\t\tWorking on generating response!")
                            try:
                                model_output = MODEL.generate(input_data=encoded_inputs)
                            except RuntimeError as e:
                                print(f"MEMORY EXCEPTION: {e}")
                                model_output = [str(e)]
                            output_dict_obj["generated-output"] = model_output

                            print("\t\tCreate path to store data!")
                            # creating track_task_output_path file json path
                            track_task_output_path = workdir.make_output_dir(
                                output_dir=self.config.output_dir,
                                model_id=model_id,
                                dataset_info=task_owl["dataset-info"],
                                encoder_id=encoder_id,
                                approach=self.approach,
                            )
                        else:
                            output_dir_path = os.path.join(
                                self.config.output_dir,
                                task_owl["dataset-info"]["track"],
                                task_owl["dataset-info"]["ontology-name"],
                            )
                            output_files_list = os.listdir(output_dir_path)
                            # 20 in the following prefix is to only get the right outputs!
                            # we need only one output per model and encoder
                            search_prefix = (
                                f"{self.approach}-{model_id}-{encoder_id}-20"
                            )
                            file_path = list(
                                filter(
                                    lambda element: element.startswith(search_prefix),
                                    output_files_list,
                                )
                            )
                            assert len(file_path) == 1
                            track_task_output_path = os.path.join(
                                output_dir_path, file_path[0]
                            )
                            output_dict_obj = io.read_json(
                                input_path=track_task_output_path
                            )
                            evaluation_results = evaluator_module(
                                track=task_owl["dataset-info"]["track"],
                                predicts=output_dict_obj["generated-output"],
                                references=task_owl["reference"],
                            )
                            output_dict_obj["evaluation-results"] = evaluation_results
                        print(f"\t\tStoring results in {track_task_output_path}.")
                        io.write_json(
                            output_path=track_task_output_path,
                            json_data=output_dict_obj,
                        )
                        print("\t\t" + "-" * 50)
                    print("\t" + "+" * 50)
