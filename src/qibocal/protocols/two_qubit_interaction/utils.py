from statistics import median_high

import numpy as np
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from ..utils import fallback_period, guess_period


def order_pair(pair: QubitPairId, platform: Platform) -> tuple[QubitId, QubitId]:
    """Order a pair of qubits by drive frequency."""
    if (
        platform.qubits[pair[0]].drive_frequency
        > platform.qubits[pair[1]].drive_frequency
    ):
        return pair[1], pair[0]
    return pair[0], pair[1]


def fit_flux_amplitude(matrix, amps, times):
    """Estimate amplitude for CZ gate.


    Given the pattern of a chevron plot (see for example Fig. 2 here
    https://arxiv.org/pdf/1907.04818.pdf). This function estimates
    the CZ amplitude by finding the amplitude which gives the highest
    oscillation period. In case there are multiple values with the same
    period, given the symmetry, the median value is chosen.
    The FFT also gives a first estimate for the duration of the CZ gate.

    Args:
     matrix (np.ndarray): signal matrix
     amps (np.ndarray): amplitudes swept
     times (np.ndarray): duration swept

    Returns:
     amplitude (float): estimated amplitude
     index (int): amplitude index
     delta (float):  omega for estimated amplitude
    """

    size_amp = len(amps)
    time_step = times[1] - times[0]
    fs = []
    for i in range(size_amp):
        y = matrix[i, :]
        period = fallback_period(guess_period(times, y))
        fs.append(1 / period)

    low_freq_interval = np.where(fs == np.min(fs))
    amplitude = median_high(amps[::-1][low_freq_interval])
    index = int(np.where(np.unique(amps) == amplitude)[0])
    delta = np.min(fs)
    return amplitude, index, delta
