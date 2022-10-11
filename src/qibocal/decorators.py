"""Decorators implementation."""
import os

from qibocal.config import raise_error


def plot(header, method):
    """Decorator for adding plots in the report and live plotting page.

    Args:
        header (str): Header of the plot to use in the report.
        method (Callable): Plotting method defined under ``qibocal.plots``.
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
