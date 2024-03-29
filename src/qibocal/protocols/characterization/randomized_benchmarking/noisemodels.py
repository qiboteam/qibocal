""" Custom error models are build here for making it possible to pass
strings describing the error model via runcards in qibocal.
They inherit from the qibo noise NoiseModel module and are prebuild.
"""

from typing import Optional

import numpy as np
from qibo import gates
from qibo.noise import NoiseModel, PauliError

from qibocal.config import raise_error


class PauliErrorOnAll(NoiseModel):
    """Builds a noise model with pauli flips
    acting on all gates in a Circuit.
    If no initial parameters for px, py, pz are given, random values
    are drawn (in sum not bigger than 1).
    """

    def __init__(self, probabilities: Optional[list] = None) -> None:
        super().__init__()
        # Check if number of arguments is 0 or 1 and if it's equal to None
        if not probabilities:
            # Assign random values to params.
            self.params = np.random.uniform(0, 0.25, size=3).round(3)
        elif len(probabilities) == 3:
            self.params = np.array(probabilities)
        else:
            # Raise ValueError if given paramters are wrong.
            raise_error(
                ValueError,
                f"Wrong number of error parameters, 3 != {len(probabilities)}.",
            )
        self.build()

    def build(self):
        # Add PauliError to gates.Gate
        self.add(PauliError(list(zip(["X", "Y", "Z"], self.params))))


class PauliErrorOnX(PauliErrorOnAll):
    """Builds a noise model with pauli flips acting on X gates.
    Inherited from ``PauliErrorOnAll`` but the ``build`` method is
    overwritten to act on X gates.
    If no initial parameters for px, py, pz are given, random values
    are drawn (in sum not bigger than 1).
    """

    def build(self):
        self.add(PauliError(list(zip(["X", "Y", "Z"], self.params))), gates.X)
