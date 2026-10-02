"""Qibo imports without warnings about CMA's optional plotting support."""

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Could not import matplotlib\.pyplot, therefore",
        category=UserWarning,
        module=r"cma\.s$",
    )
    from qibo import Circuit, gates

__all__ = ["Circuit", "gates"]
