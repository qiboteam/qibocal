""" Costum error models are build here for making it possible to pass
strings describing the error model via runcards in qibocal.
They inherit from the qibo noise NoiseModel module and are prebuild.
"""

import numpy as np
from qibo import gates
from qibo.noise import *
from qibo.hamiltonians import Hamiltonian

from qibocal.config import raise_error


class PauliErrorOnAll(NoiseModel):
    """Builds a noise model with pauli flips
    acting on all gates in a Circuit.

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
        # Add PauliError to gates.Gate
        self.add(PauliError(*params))


class PauliErrorOnX(PauliErrorOnAll):
    """Builds a noise model with pauli flips acting on X gates.

    Inherited from ``PauliErrorOnAll`` but the ``build`` method is
    overwritten to act on X gates.
    If no initial parameters for px, py, pz are given, random values
    are drawn (in sum not bigger than 1).
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def build(self, *params):
        self.add(PauliError(*params), gates.X)


class PauliErrorOnXAndRX(PauliErrorOnAll):
    """Builds a noise model with pauli flips acting on X and RX gates.

    Inherited from ``PauliErrorOnAll`` but the ``build`` method is
    overwritten to act on X and RX gates.
    If no initial parameters for px, py, pz are given, random values
    are drawn (in sum not bigger than 1).
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def build(self, *params):
        self.add(PauliError(*params), gates.X)
        self.add(PauliError(*params), gates.RX)


class PauliErrorOnNonDiagonal(PauliErrorOnAll):
    """Builds a noise model with pauli flips acting on gates X, Y and Unitary that are not diagonal.

    Inherited from ``PauliErrorOnAll`` but the ``build`` method is
    overwritten to act on X and Y and Unitary gates.
    If no initial parameters for px, py, pz are given, random values
    are drawn (in sum not bigger than 1).
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def build(self, *params):
        is_non_diag = lambda g: not np.allclose(np.abs(g.parameters), np.eye(2))
        self.add(PauliError(*params), gates.X)
        self.add(PauliError(*params), gates.Y)
        self.add(PauliError(*params), gates.Unitary, condition=is_non_diag)


class UnitaryErrorOnAll(NoiseModel):
    """Builds a noise model with a unitary error
    acting on all gates in a Circuit.

    If matrix ``U`` is not given,
    a random unitary close to identity is generated.
    """

    def __init__(self, probabilities=[], unitaries=[], nqubits=1, t=0.1) -> None:
        super().__init__()

        # If unitaries are not given, generate a random Unitary close to Id
        if len(unitaries) == 0:
            from qibo.quantum_info import random_hermitian
            from scipy.linalg import expm

            dim = 2**nqubits

            # Generate random unitary matrix close to Id. U=exp(i*l*G)
            herm_generator = random_hermitian(dim)
            unitary_matr = expm(-1j * t * herm_generator)

            self.unitaries = [unitary_matr]
            self.probabilities = [1]
        else:
            self.probabilities = probabilities
            self.unitaries = unitaries

        self.build()

    def build(self):
        self.add(UnitaryError(self.probabilities, self.unitaries))


class UnitaryErrorOnX(UnitaryErrorOnAll):
    """Builds a noise model with a unitary error
    acting on all gates in a Circuit.

    Inherited from ``UnitaryErrorOnAll`` but the ``build`` method is
    overwritten to act on X gates.
    If matrix ``U`` is not given,
    a random unitary close to identity is generated.
    """

    def __init__(self, probabilities=[], unitaries=[], nqubits=1, t=0.1) -> None:
        super().__init__()

        # If unitaries are not given, generate a random Unitary close to Id
        if len(unitaries) == 0:
            from qibo.quantum_info import random_hermitian
            from scipy.linalg import expm

            dim = 2**nqubits

            # Generate random unitary matrix close to Id. U=exp(i*l*G)
            herm_generator = random_hermitian(dim)
            unitary_matr = expm(-1j * t * herm_generator)

            self.unitaries = [unitary_matr]
            self.probabilities = [1]
        else:
            self.probabilities = probabilities
            self.unitaries = unitaries

        self.build()

    def build(self):
        self.add(UnitaryError(self.probabilities, self.unitaries), gates.X)


class ThermalRelaxationErrorOnUnitary(NoiseModel):
    """Builds a noise model with thermal relaxation error acting on unitaries."""

    def __init__(self, *args) -> None:
        super().__init__()
        # Check if number of arguments is 3 or 4
        if len(args) == 3 or len(args) == 4:
            params = args
        else:
            # Raise ValueError if given paramters are wrong.
            raise_error(
                ValueError,
                "Wrong number of error parameters, {} instead of 3 or 4.".format(
                    len(args)
                ),
            )
        self.build(*params)

    def build(self, *params):
        # Add ThermalRelaxationError to gates.Unitary
        self.add(ThermalRelaxationError(*params), gates.Unitary)


class ThermalRelaxationErrorOnX(ThermalRelaxationErrorOnUnitary):
    """Builds a noise model with thermal relaxation error acting on X gates."""

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def build(self, *params):
        # Add ThermalRelaxationError to gates.X
        self.add(ThermalRelaxationError(*params), gates.X)


class ThermalRelaxationErrorOnXAndRX(ThermalRelaxationErrorOnUnitary):
    """Builds a noise model with thermal relaxation error acting on X and RX gates."""

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def build(self, *params):
        self.add(ThermalRelaxationError(*params), gates.X)
        self.add(ThermalRelaxationError(*params), gates.RX)


class ThermalRelaxationErrorOnNonDiagonal(ThermalRelaxationErrorOnUnitary):
    """Builds a noise model with thermal relaxation error acting on X, Y and non-diagonal Unitary gates."""

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def build(self, *params):
        is_non_diag = lambda g: not np.allclose(np.abs(g.parameters), np.eye(2))
        self.add(ThermalRelaxationError(*params), gates.X)
        self.add(ThermalRelaxationError(*params), gates.Y)
        self.add(ThermalRelaxationError(*params), gates.Unitary, condition=is_non_diag)
