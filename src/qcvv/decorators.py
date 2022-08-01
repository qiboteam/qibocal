# -*- coding: utf-8 -*-
import os


def prepare_path(name=None, folder=None):
    """Helper function to create the output folder for a particular calibration."""
    path = os.path.join(folder, f"data/{name}/")
    os.makedirs(path)
    return path


def save(results, path, format="csv"):
    """Helper function to store the data in a particular format."""
    for data in results:
        getattr(data, f"to_{format}")(path)


def store(f):
    """Decorator storing data."""
    f.prepare = prepare_path
    f.final_action = save
    return f
