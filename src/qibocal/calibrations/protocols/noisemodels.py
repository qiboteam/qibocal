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
        if len(args) == 0 or (len(args) == 1 and args[0] is None):
            params = np.random.uniform(0, 1, size=3)
            if sum(params) > 1.0:
                params /= sum(params)
        elif len(args) == 3:
            params = args
        else:
            raise_error(
                ValueError,
                "Wrong number of error parameters, 3 != {}.".format(len(args)),
            )
        self.build(*params)

    def build(self, *params):
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


class LizasNoiseModel(NoiseModel):
    pass
