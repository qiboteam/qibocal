# -*- coding: utf-8 -*-
"""Decorators implementation."""
import os

from qcvv.config import raise_error


def plot(header, method, qubit_sections=False):
    """Decorator for adding plots in the report and live plotting page.

    Args:
        header (str): Header of the plot to use in the report.
        method (Callable): Plotting method defined under ``qcvv.plots``.
        qubit_sections (bool): If ``True`` each qubit is plotted in a different HTML section.
            Default is ``False``.
    """

    def wrapped(f):
        if hasattr(f, "plots"):
            # insert in the beginning of the list to have
            # proper plot ordering in the report
            f.plots.insert(0, (header, method, qubit_sections))
        else:
            f.plots = [(header, method, qubit_sections)]
        return f

    return wrapped
