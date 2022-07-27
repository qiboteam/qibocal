# -*- coding: utf-8 -*-
import os

import yaml


def prepare_path(name=None, folder=None):
    path = os.path.join(folder, f"data/{name}/")
    os.makedirs(path)
    return path


def save(results, path, format="yaml"):
    for data in results:
        output = {}
        for i, c in data.container.items():
            output[i] = c.to_dict()
        with open(f"{path}/data.yml", "w") as f:
            yaml.dump(output, f)


def store(f):
    f.prepare = prepare_path
    f.final_action = save
    return f
