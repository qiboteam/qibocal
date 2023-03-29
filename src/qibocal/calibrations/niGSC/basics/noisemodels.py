""" Costum error models are build here for making it possible to pass
strings describing the error model via runcards in qibocal.
They inherit from the qibo noise NoiseModel module and are prebuild.
"""

import numpy as np
from qibo import gates
from qibo.hamiltonians import Hamiltonian
from qibo.noise import *

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

    If parameters are not given,
    a random unitary close to identity is generated
    ::math:`U = \\exp(-i t H)` for a random Harmitian ::math:`H`.

    Args:
        probabilities (list): list of probabilities corresponding to unitaries. Defualt is [].
        unitaries (list): list of unitaries. Defualt is [].
        nqubits (int): number of qubits. Default is 1.
        t (float): "strength" of random unitary noise. Default t=0.1.
    """

    def __init__(self, probabilities=[], unitaries=[], nqubits=1, t=0.1) -> None:
        super().__init__()

        if not isinstance(t, float):
            raise_error(
                TypeError, f"Parameter t must be of type float. Got {type(t)} instead."
            )
        # If unitaries are not given, generate a random Unitary close to Id
        if len(unitaries) == 0:
            from qibo.quantum_info import random_hermitian
            from scipy.linalg import expm

            dim = 2**nqubits

            # Generate random unitary matrix close to Id. U=exp(i*t*H)
            herm_generator = random_hermitian(dim)
            unitary_matr = expm(-1j * t * herm_generator)

            unitaries = [unitary_matr]
            probabilities = [1]

        self.build(probabilities, unitaries)

    def build(self, probabilities, unitaries):
        self.add(UnitaryError(probabilities, unitaries))


class UnitaryErrorOnX(UnitaryErrorOnAll):
    """Builds a noise model with a unitary error
    acting on all gates in a Circuit.

    Inherited from ``UnitaryErrorOnAll`` but the ``build`` method is
    overwritten to act on X gates.
    If matrix ``U`` is not given,
    a random unitary close to identity is generated.
    """

    def __init__(self, probabilities=[], unitaries=[], nqubits=1, t=0.1) -> None:
        super().__init__(probabilities, unitaries, nqubits, t)

    def build(self, probabilities, unitaries):
        self.add(UnitaryError(probabilities, unitaries), gates.X)


class UnitaryErrorOnXAndRX(UnitaryErrorOnAll):
    """Builds a noise model with a unitary error
    acting on all gates in a Circuit.

    Inherited from ``UnitaryErrorOnAll`` but the ``build`` method is
    overwritten to act on X and RX gates.
    If matrix ``U`` is not given,
    a random unitary close to identity is generated.
    """

    def __init__(self, probabilities=[], unitaries=[], nqubits=1, t=0.1) -> None:
        super().__init__(probabilities, unitaries, nqubits, t)

    def build(self, probabilities, unitaries):
        self.add(UnitaryError(probabilities, unitaries), gates.X)
        self.add(UnitaryError(probabilities, unitaries), gates.RX)


class UnitaryErrorOnNonDiagonal(UnitaryErrorOnAll):
    """Builds a noise model with a unitary error
    acting on all gates in a Circuit.

    Inherited from ``UnitaryErrorOnAll`` but the ``build`` method is
    overwritten to act on on gates X, Y and non-diagonal Unitary gates.
    If matrix ``U`` is not given,
    a random unitary close to identity is generated.
    """

    def __init__(self, probabilities=[], unitaries=[], nqubits=1, t=0.1) -> None:
        super().__init__(probabilities, unitaries, nqubits, t)

    def build(self, probabilities, unitaries):
        is_non_diag = lambda g: not np.allclose(np.abs(g.parameters), np.eye(2))
        self.add(UnitaryError(probabilities, unitaries), gates.X)
        self.add(UnitaryError(probabilities, unitaries), gates.Y)
        self.add(
            UnitaryError(probabilities, unitaries), gates.Unitary, condition=is_non_diag
        )


class ThermalRelaxationErrorOnAll(NoiseModel):
    """Builds a noise model with thermal relaxation error acting on all gates in a circuit."""

    def __init__(self, *args) -> None:
        super().__init__()
        # Check if number of arguments is 0 or 1 and if it's equal to None
        if len(args) == 0 or (len(args) == 1 and args[0] is None):
            t1 = np.random.uniform(0.0, 10.0)
            t2 = np.random.uniform(0.0, 2 * t1)
            coeff = np.random.uniform(1.0, 10.0)
            time = t1 / coeff
            a0 = np.random.uniform(0.0, 1.0)
            params = [t1, t2, time, a0]
        # Check if number of arguments is 3 or 4
        elif len(args) == 3 or len(args) == 4:
            params = args
        else:
            # Raise ValueError if given paramters are wrong.
            raise_error(
                ValueError,
                f"Wrong number of error parameters: {len(args)} instead of 3 or 4.",
            )
        self.build(*params)

    def build(self, *params):
        # Add ThermalRelaxationError to gates.Unitary
        self.add(ThermalRelaxationError(*params))


class ThermalRelaxationErrorOnX(ThermalRelaxationErrorOnAll):
    """Builds a noise model with thermal relaxation error acting on X gates."""

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def build(self, *params):
        # Add ThermalRelaxationError to gates.X
        self.add(ThermalRelaxationError(*params), gates.X)


class ThermalRelaxationErrorOnXAndRX(ThermalRelaxationErrorOnAll):
    """Builds a noise model with thermal relaxation error acting on X and RX gates."""

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def build(self, *params):
        self.add(ThermalRelaxationError(*params), gates.X)
        self.add(ThermalRelaxationError(*params), gates.RX)


class ThermalRelaxationErrorOnNonDiagonal(ThermalRelaxationErrorOnAll):
    """Builds a noise model with thermal relaxation error acting on X, Y and non-diagonal Unitary gates."""

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def build(self, *params):
        is_non_diag = lambda g: not np.allclose(np.abs(g.parameters), np.eye(2))
        self.add(ThermalRelaxationError(*params), gates.X)
        self.add(ThermalRelaxationError(*params), gates.Y)
        self.add(ThermalRelaxationError(*params), gates.Unitary, condition=is_non_diag)
