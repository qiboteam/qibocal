import numpy as np


def fit_Z_exp(t, wx, wy, wz, w):
    """Fitting Z expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    return ((wx**2 + wy**2) * np.cos(w * t) + wz**2) / w**2


def fit_Z_exp_fine(t, wx, wy, wz):
    """Fitting Z expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    w = np.sqrt(wx**2 + wy**2 + wz**2)
    return ((wx**2 + wy**2) * np.cos(w * t) + wz**2) / w**2


def fit_X_exp(t, wx, wy, wz, w):
    """Fitting X expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    return (-wx * wz * np.cos(w * t) + w * wy * np.sin(t * w) + wx * wz) / w**2


def fit_X_exp_fine(t, wx, wy, wz):
    """Fitting X expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    w = np.sqrt(wx**2 + wy**2 + wz**2)
    return (-wx * wz * np.cos(w * t) + w * wy * np.sin(t * w) + wx * wz) / w**2


def fit_Y_exp(t, wx, wy, wz, w):
    """Fitting Y expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    return (-w * wx * np.sin(w * t) - wy * wz * np.cos(t * w) + wy * wz) / w**2


def fit_Y_exp_fine(t, wx, wy, wz):
    """Fitting Y expectation value for CR tomography.

    See https://arxiv.org/pdf/2303.01427 Eq. S10.
    """
    w = np.sqrt(wx**2 + wy**2 + wz**2)
    return (-w * wx * np.sin(w * t) - wy * wz * np.cos(t * w) + wy * wz) / w**2


def combined_fit(x, wx, wy, wz):
    """Combined fit for CR tomography."""
    w = np.sqrt(wx**2 + wy**2 + wz**2)
    x1, x2, x3 = np.split(x, 3)
    return np.concatenate(
        [
            fit_X_exp(x1, wx, wy, wz, w),
            fit_Y_exp(x2, wx, wy, wz, w),
            fit_Z_exp(x3, wx, wy, wz, w),
        ]
    )
