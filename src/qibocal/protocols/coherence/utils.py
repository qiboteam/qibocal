"""Backward-compatibility shim.

New code should import from the themed modules directly:

- ``qibocal.protocols.coherence.acquisition``
- ``qibocal.protocols.coherence.fitting``
- ``qibocal.protocols.coherence.plotting``
"""

from .acquisition import (
    CoherenceType,
    average_single_shots,
    dynamical_decoupling_sequence,
)
from .fitting import (
    exp_decay,
    exponential_fit,
    exponential_fit_probability,
    single_exponential_fit,
)
from .plotting import plot

__all__ = [
    "CoherenceType",
    "average_single_shots",
    "dynamical_decoupling_sequence",
    "exp_decay",
    "exponential_fit",
    "exponential_fit_probability",
    "single_exponential_fit",
    "plot",
]
