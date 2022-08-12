# -*- coding: utf-8 -*-
"""Decorators implementation."""
import os

from qcvv.config import raise_error


def prepare_path(name=None, folder=None):
    """Helper function to create the output folder for a particular calibration."""
    path = os.path.join(folder, f"data/{name}/")
    os.makedirs(path)
    return path


def save(results, path, format=None):
    """Helper function to store the data in a particular format."""

    if format is None:
        raise_error(ValueError, f"Cannot store data using {format} format.")

    for data in results:
        getattr(data, f"to_{format}")(path)


def fitting_routine(routine, folder, format, platform, qubit, params):
    """Helper function to call the fitting methods"""
    from qcvv.fitting import methods

    params = getattr(methods, f"{routine.__name__}_fit")(
        folder, format, platform, qubit, params
    )
    return params


def update_routine(routine, folder, qubit, data):
    """Helper function to update the platform runcard."""
    from qcvv.fitting import update

    getattr(update, f"{routine.__name__}_update")(folder, qubit, *data)


def store(f):
    """Decorator for storing data."""
    f.prepare = prepare_path
    f.final_action = save
    return f


def fit(f):
    """Decorator fitting data and updating the platform runcard."""
    f.post_processing = fitting_routine
    f.update = update_routine
    return f
