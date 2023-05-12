""" Costum error models are build here for making it possible to pass
strings describing the error model via runcards in qibocal.
They inherit from the qibo noise NoiseModel module and are prebuild.
"""

import numpy as np
from qibo import gates
from qibo.noise import NoiseModel, PauliError

from qibocal.config import raise_error


class PauliErrorOnUnitary(NoiseModel):
    """Builds a noise model with pauli flips acting on unitaries.

    If no initial parameters for px, py, pz are given, random values
    are drawn (in sum not bigger than 1).
    """

    def __init__(self, *args) -> None:
        super().__init__()
        # Check if number of arguments is 0 or 1 and if it's equal to None
        if len(args) == 0 or (len(args) == 1 and args[0] is None):
            # Assign random values to params.
            params = np.random.uniform(0, 0.25, size=3)
        elif len(args) == 3:
            params = args
        else:
            # Raise ValueError if given paramters are wrong.
            raise_error(
                ValueError,
                "Wrong number of error parameters, 3 != {}.".format(len(args)),
            )
        self.build(*params)

    def build(self, *params):
        # TODO for qibo v.0.1.14 change *params to list(zip(["X", "Y", "Z"], params))
        # Add PauliError to gates.Unitary
        self.add(PauliError(*params), gates.Unitary)


class PauliErrorOnX(PauliErrorOnUnitary):
    """Builds a noise model with pauli flips acting on X gates.

    Inherited from ``PauliErrorOnUnitary`` but the ``build`` method is
    overwritten to act on X gates.
    If no initial parameters for px, py, pz are given, random values
    are drawn (in sum not bigger than 1).
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def build(self, *params):
        self.add(PauliError(*params), gates.X)
