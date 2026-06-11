"""Fitting function for CR tomography."""

import numpy as np


def pauli_z_expectation(
    t: np.typing.NDArray, wx: float, wy: float, wz: float, gamma: float
) -> np.typing.NDArray:
    """Pauli Z expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    return vectorized_simultaneous_expectation(t, wx, wy, wz, gamma).reshape(3, -1)[
        2, :
    ]


def pauli_x_expectation(
    t: np.typing.NDArray, wx: float, wy: float, wz: float, gamma: float
) -> np.typing.NDArray:
    """Pauli X expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    return vectorized_simultaneous_expectation(t, wx, wy, wz, gamma)[0, :]


def pauli_y_expectation(
    t: np.typing.NDArray, wx: float, wy: float, wz: float, gamma: float
) -> np.typing.NDArray:
    """Pauli Y expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    return vectorized_simultaneous_expectation(t, wx, wy, wz, gamma)[1, :]


def vectorized_simultaneous_expectation(
    t: np.typing.NDArray,
    wx: float,
    wy: float,
    wz: float,
    gamma: float,
) -> np.typing.NDArray:

    # Compute the norm of the w vector with components wx, wy, wz
    w = np.sqrt(wx**2 + wy**2 + wz**2)

    # Map cosine, sine, and constant time components to X, Y, Z expectation values.
    system_matrix = np.array(
        [
            [-wx * wz, w * wy, wx * wz],
            [-wy * wz, -w * wx, wy * wz],
            [wx**2 + wy**2, 0, wz**2],
        ]
    )

    # Create time array with cosine, sine, and constant components
    time_array = np.array([np.cos(t * w), np.sin(t * w), np.ones_like(t)])

    # Apply system matrix transformation, multiply by exponential decay, normalize
    return (system_matrix @ time_array) * np.exp(-gamma * t) / w**2


def simultaneous_expectation(
    t: np.typing.NDArray,
    wx: float,
    wy: float,
    wz: float,
    gamma: float,
) -> np.typing.NDArray:
    """Simulateneous computation for X, Y and Z expectation values.

    We also constrain w to be the norm of the w vector with component
    wx, wy, wz.
    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """

    # flatten the vectorized solution, necessary for curve_fit
    return vectorized_simultaneous_expectation(t, wx, wy, wz, gamma).ravel()


def linear_func(x, a, b):
    return a * x + b


def sin_func(x, a, b, omega, phi):
    return a * np.sin(x * omega + phi) + b
