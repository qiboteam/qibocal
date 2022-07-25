# -*- coding: utf-8 -*-
import os
import time

import yaml


def generate_output(folder=None):
    def inner(func):
        def wrapper(*args, **kwargs):
            path = os.path.join(
                folder, f"{func.__name__}/{time.strftime('%Y%m%d-%H%M%S')}"
            )
            os.makedirs(path)
            for data in func(*args, **kwargs):
                output = {}
                for i, c in data.container.items():
                    output[i] = c.to_dict()
                with open(f"{path}/data.yml", "w") as f:
                    yaml.dump(output, f)

        return wrapper

    return inner
