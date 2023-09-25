from statistics import median_high

import numpy as np
from qibolab.qubits import Qubit, QubitId
from scipy.signal import find_peaks

RANDOM_HIGH_VALUE = 1e6
"""High value to avoid None when computing FFT."""


def order_pair(pair: list[QubitId, QubitId], qubits: dict[QubitId, Qubit]) -> tuple:
    """Order a pair of qubits by drive frequency."""
    if qubits[pair[0]].drive_frequency > qubits[pair[1]].drive_frequency:
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
     matrix (np.ndarray): msr matrix
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
        ft = np.fft.rfft(y) / len(y)
        mags = abs(ft)[1:]
        local_maxima = find_peaks(mags, height=0)
        peak_heights = local_maxima[1]["peak_heights"]
        # Select the frequency with the highest peak
        index = (
            int(local_maxima[0][np.argmax(peak_heights)] + 1)
            if len(local_maxima[0]) > 0
            else None
        )

        sampling_freq = 1 / time_step
        values = np.arange(int(len(y) / 2))
        period = len(y) / sampling_freq
        frequencies = values / period
        f = frequencies[index] if index is not None else RANDOM_HIGH_VALUE
        fs.append(2 * np.pi * f)

    low_freq_interval = np.where(fs == np.min(fs))
    amplitude = median_high(amps[low_freq_interval])
    index = int(np.where(np.unique(amps) == amplitude)[0])
    delta = np.min(fs)
    return amplitude, index, delta
