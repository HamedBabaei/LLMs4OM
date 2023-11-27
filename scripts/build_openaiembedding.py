# -*- coding: utf-8 -*-
import datetime
import os

import numpy as np
import openai
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from tqdm import tqdm

from ontomap.base import BaseConfig
from ontomap.ontology import ontology_matching
from ontomap.tools import workdir
from ontomap.utils import io

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_KEY"]
config = BaseConfig(approach="lightweight").get_args(device="cpu")
client = OpenAI(api_key=openai.api_key)


def get_embedding(text, model="text-embedding-ada-002"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def preprocess(text: str) -> str:
    text = text.replace("_", " ")
    text = text.lower()
    return text


if __name__ == "__main__":
    label_set = []
    for track, tasks in ontology_matching.items():
        for task in tasks:
            task_obj = task()
            print(f"Working on {task_obj} task")
            task_owl = task_obj.load_from_json(root_dir=config.root_dir)
            for owl in task_owl["target"]:
                label_set.append(preprocess(owl["label"]))
            for owl in task_owl["source"]:
                label_set.append(preprocess(owl["label"]))

    print("Total size:", len(label_set))
    print("Total unique labels:", len(list(set(label_set))))

    embeddings = []
    labels2index = {}
    labels = list(set(label_set))

    for index, label in tqdm(enumerate(labels)):
        labels2index[label] = index
        if label != "":
            embedding = get_embedding(label)
        else:
            embedding = get_embedding(" ")
        embeddings.append(embedding)

    matrix = np.array(embeddings)

    workdir.mkdir(path=config.openai_embedding_dir)

    np.save(
        os.path.join(config.openai_embedding_dir, "openai_embeddings.npy"), embeddings
    )

    io.write_json(
        output_path=os.path.join(config.openai_embedding_dir, "labels2index.json"),
        json_data=labels2index,
    )
    model_info = {
        "model-name": "text-embedding-ada-002",
        "input-representation-of": "labels",
        "creation-date": str(datetime.datetime()),
        "examples-len": str(len(label_set)),
        "examples-unique-len": str(len(labels)),
        "embedding-dim": f"({str(matrix.shape[0])}, {str(matrix.shape[1])})",
        "index2labels-size": str(len(labels2index)),
    }

    io.write_json(
        output_path=os.path.join(config.openai_embedding_dir, "model-info.json"),
        json_data=model_info,
    )
