# -*- coding: utf-8 -*-
import os
from typing import Dict, List


from ontomap.base import BaseConfig
from ontomap.ontology import ontology_matching
from ontomap.utils import io


def calculate_key_counts(list_of_items: List, key_to_calculate_stat: str) -> int:
    count = 0
    for item in list_of_items:
        if len(item[key_to_calculate_stat]) != 0:
            count += 1
    return count


def measure_ontologies_statistics(parsed_ontology: List) -> Dict:
    statistics = {
        "size": len(parsed_ontology),
        "childrens": calculate_key_counts(
            list_of_items=parsed_ontology, key_to_calculate_stat="childrens"
        ),
        "parents": calculate_key_counts(
            list_of_items=parsed_ontology, key_to_calculate_stat="parents"
        ),
        "comment": calculate_key_counts(
            list_of_items=parsed_ontology, key_to_calculate_stat="comment"
        ),
        "synonyms": calculate_key_counts(
            list_of_items=parsed_ontology, key_to_calculate_stat="synonyms"
        ),
    }
    return statistics


def measure_statistics(dataset: Dict) -> Dict:
    track = dataset["dataset-info"]["track"]
    statistic = {
        "task": dataset["dataset-info"]["ontology-name"],
        "source": measure_ontologies_statistics(parsed_ontology=dataset["source"]),
        "target": measure_ontologies_statistics(parsed_ontology=dataset["target"]),
    }
    if track == "bio-llm":
        equiv, subs = len(dataset["reference"]["test-cands"]), 0
        reference_size = equiv + subs
    elif track == "bio-ml":
        equiv, subs = len(dataset["reference"]["equiv"]["full"]), len(
            dataset["reference"]["subs"]["test-cands"]
        ) + len(dataset["reference"]["subs"]["train"])
        reference_size = equiv + subs
    else:
        equiv, subs = 0, 0
        for item in dataset["reference"]:
            if item["relation"] == "=":
                equiv += 1
            elif item["relation"] == ">" or item["relation"] == "<":
                subs += 1
        reference_size = len(dataset["reference"])
    assert reference_size == equiv + subs
    statistic["reference"] = {
        "size": reference_size,
        "equiv": equiv,
        "subs": subs,
    }
    return statistic


def convert_dataset_stats_to_latax(statistics: Dict, latax_path: str):
    refactorers = {
        "taxrefldBacteria-ncbitaxonBacteria": "taxrefld-ncbi (Bacteria)",
        "taxrefldChromista-ncbitaxonChromista": "taxrefld-ncbi (Chromista)",
        "taxrefldPlantae-ncbitaxonPlantae": "taxrefld-ncbi (Plantae)",
        "taxrefldFungi-ncbitaxonFungi": "taxrefld-ncbi (Fungi)",
        "taxrefldProtozoa-ncbitaxonProtozoa": "taxrefld-ncbi (Protozoa)",
        "MaterialInformation-EMMO": "MI-EMMO",
        "MaterialInformation-MatOnto": "MI-MatOnto",
        "MaterialInformationReduced-MatOnto": "MIReduced-MatOnto",
        "macroalgae-macrozoobenthos": "algae-zoobenthos (Macro)",
    }
    file = open(latax_path, "w", encoding="utf-8")
    file.write(
        """
    \\begin{table}
        \caption{OAEI tracks and tasks statistics across source, target, and alignments.}\label{tab_dataset_stats}
        \\begin{tabular}{|l|l|c|c|c|c|c|c|c|c|c|c|c|}
            \hline
            \multirow{1}{*}{\\rotatebox{90}{\\textbf{Track}}} & \multirow{2}{*}{\\textbf{Task}} & \multicolumn{2}{|c|}{\\textbf{Clss}} &
            \multicolumn{2}{|c|}{\\textbf{Childs}} & \multicolumn{2}{|c|}{\\textbf{Parents}} &
            \multicolumn{2}{|c|}{\\textbf{Cmt}} & \multicolumn{3}{|c|}{\\textbf{Alig}}\\\\
            \cline{3-13}
             &  & \multirow{2}{*}{S} & \multirow{2}{*}{T} & \multirow{2}{*}{S} & \multirow{2}{*}{T} & \multirow{2}{*}{S} & \multirow{2}{*}{T} &  \multirow{2}{*}{S} & \multirow{2}{*}{T} & \multirow{2}{*}{Eqv} & \multirow{2}{*}{Sub} & \multirow{2}{*}{All}\\\\
             & & & &  && & & &  &  & & \\\\
            \hline

    """
    )
    file.write("\n")

    for track, track_items in statistics.items():
        row_no = len(track_items) if len(track_items) > 4 else 4
        row_counter = len(track_items)
        track_prefix = (
            "\multirow{"
            + str(row_no)
            + "}{*}{\\rotatebox{90}{\\textit{"
            + track
            + "}}}"
        )
        first_row = True
        for task in track_items:
            row = (
                f"& {refactorers.get(task['task'], task['task'])} & "
                + f"{task['source']['size']}&{task['target']['size']} &"
                + f"{task['source']['childrens']}&{task['target']['childrens']} & "
                + f"{task['source']['parents']}&{task['target']['parents']} & "
                + f"{task['source']['comment']}&{task['target']['comment']} & "
                + f"{task['reference']['equiv']}&{task['reference']['subs']} & {task['reference']['size']}"
            )
            if first_row:
                row = track_prefix + row
                first_row = False
            else:
                row = row
            row += "\\\\"
            # file.write("\n")
            file.write(row)
            file.write("\n")

        while row_counter < row_no:
            file.write("& & & &  && & & &  &  & & \\\\")
            file.write("\n")
            row_counter += 1
        file.write("\hline")

    file.write(
        """
        \end{tabular}
    \end{table}
    """
    )
    file.close()


if __name__ == "__main__":
    config = BaseConfig().get_args()

    statistics = {}
    for ontology, oms in ontology_matching.items():
        print(f"working on {ontology} track OM pairs!")
        for om in oms:
            print(f"\t {om.ontology_name} pairs is processing!")
            dataset = om().collect(root_dir=config.root_dir)
            if dataset["dataset-info"]["track"] not in statistics:
                statistics[dataset["dataset-info"]["track"]] = []
            stats = measure_statistics(dataset)
            statistics[dataset["dataset-info"]["track"]].append(stats)

    io.write_json(
        output_path=os.path.join(config.stats_dir, "tracks-tasks-stats.json"),
        json_data=statistics,
    )
    latax_path = os.path.join(config.stats_dir, "tracks-tasks-stats.tex")
    convert_dataset_stats_to_latax(statistics, latax_path)
