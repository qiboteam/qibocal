from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data

RBType = np.dtype(
    [
        ("samples", np.float64),
    ]
)
"""Custom dtype for RB."""


@dataclass
class RBData(Data):
    """A pandas DataFrame bastard child. The output of the acquisition function."""

    params: dict
    depths: list
    data: dict[QubitId, npt.NDArray[RBType]] = field(default_factory=dict)
    """Raw data acquired."""
