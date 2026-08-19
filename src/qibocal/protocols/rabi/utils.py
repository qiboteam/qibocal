"""Backward-compatibility shim.

The functions previously defined here moved to the themed modules
:mod:`fitting`, :mod:`plotting` and :mod:`acquisition`.
"""

from .acquisition import sequence_amplitude, sequence_length
from .fitting import (
    QUANTILE_CONSTANT_RABI,
    extract_rabi,
    fit_amplitude_function,
    fit_length_function,
    period_correction_factor,
    rabi_amplitude_function,
    rabi_initial_guess,
    rabi_length_function,
)
from .plotting import plot, plot_probabilities

__all__ = [
    "QUANTILE_CONSTANT_RABI",
    "extract_rabi",
    "fit_amplitude_function",
    "fit_length_function",
    "period_correction_factor",
    "plot",
    "plot_probabilities",
    "rabi_amplitude_function",
    "rabi_initial_guess",
    "rabi_length_function",
    "sequence_amplitude",
    "sequence_length",
]
