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


def store(f):
    """Decorator storing data."""
    f.prepare = prepare_path
    f.final_action = save
    return f


def plot(header, method):
    """Decorator for adding plots in the report and live plotting page.

    Args:
        header (str): Header of the plot to use in the report.
        method (Callable): Plotting method defined under ``qcvv.plots``.
    """

    def wrapped(f):
        if hasattr(f, "plots"):
            # insert in the beginning of the list to have
            # proper plot ordering in the report
            f.plots.insert(0, (header, method))
        else:
            f.plots = [(header, method)]
        return f

    return wrapped
