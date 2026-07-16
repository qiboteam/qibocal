from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
)

HamiltonianTomographyType = np.dtype(
    [
        ("prob_target", np.float64),
        ("error_target", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
        ("x", np.float64),
    ]
)
"""Custom dtype for Hamiltonian Tomography."""


class SetControl(str, Enum):
    """Helper to create sequence with control set to X or I."""

    Id = "Id"
    X = "X"


class Basis(str, Enum):
    """Measurement basis."""

    X = "X"
    Y = "Y"
    Z = "Z"


class HamiltonianTerm(str, Enum):
    """Hamiltonian terms for CR effective Hamiltonian."""

    IX = "IX"
    IY = "IY"
    IZ = "IZ"
    ZX = "ZX"
    ZY = "ZY"
    ZZ = "ZZ"


@dataclass
class HamiltonianTomographyParameters(Parameters):
    """Parent class for parameters in all time-sweeping hamiltonian tomography experiments."""

    duration_range: tuple[float, float, float]
    """Range of amplitudes for CR pulse (start, end, step)."""
    echo: bool = False
    """Apply echo sequence or not.

    The ECR is described in https://arxiv.org/pdf/1210.7011
    """
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""


@dataclass
class HamiltonianTomographyResults(Results):
    """Results for Hamiltonian Tomography CR Length experiment."""

    echo: bool
    cr_lengths: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """Estimated durations of CR gate."""
    hamiltonian_terms: dict = field(default_factory=dict)
    """Terms in effective Hamiltonian."""
    fitted_parameters: dict = field(default_factory=dict)
    """Fitted parameters from X,Y,Z expectation values."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class HamiltonianTomographyData(Data):
    """Data for Hamiltonian Tomography CR Amplitude experiment."""

    echo: bool
    data: dict[tuple[QubitId, QubitId, Basis, SetControl], Any] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    @property
    def pairs(self):
        return {(i[0], i[1]) for i in self.data}
