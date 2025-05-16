"""Fitting function for CR tomography."""

import numpy as np


def fit_Z_exp(t: np.ndarray, wx: float, wy: float, wz: float, w: float) -> np.ndarray:
    """Fitting Z expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    return ((wx**2 + wy**2) * np.cos(w * t) + wz**2) / w**2


def fit_X_exp(t: np.ndarray, wx: float, wy: float, wz: float, w: float) -> np.ndarray:
    """Fitting X expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    return (-wx * wz * np.cos(w * t) + w * wy * np.sin(t * w) + wx * wz) / w**2


def fit_Y_exp(t: np.ndarray, wx: float, wy: float, wz: float, w: float) -> np.ndarray:
    """Fitting Y expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    return (-w * wx * np.sin(w * t) - wy * wz * np.cos(t * w) + wy * wz) / w**2


def combined_fit(
    t: np.ndarray,
    wx: float,
    wy: float,
    wz: float,
) -> np.ndarray:
    """Simulateneous fit for X, Y and Z expectation values.

    We also constrain w to be the norm of the w vector with component
    wx, wy, wz."""

    w = np.sqrt(wx**2 + wy**2 + wz**2)
    t1, t2, t3 = np.split(t, 3)
    return np.concatenate(
        [
            fit_X_exp(t1, wx, wy, wz, w),
            fit_Y_exp(t2, wx, wy, wz, w),
            fit_Z_exp(t3, wx, wy, wz, w),
        ]
    )
