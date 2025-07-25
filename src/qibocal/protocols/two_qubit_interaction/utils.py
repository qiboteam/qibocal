import numpy as np
from qibolab import Platform
from scipy.optimize import curve_fit

from qibocal.auto.operation import QubitId, QubitPairId
from qibocal.config import log

from ..utils import fallback_period, guess_period


def order_pair(pair: QubitPairId, platform: Platform) -> tuple[QubitId, QubitId]:
    """Order a pair of qubits by drive frequency."""
    q0, q1 = pair

    drive0 = platform.config(platform.qubits[q0].drive)
    drive1 = platform.config(platform.qubits[q1].drive)
    return (q1, q0) if drive0.frequency > drive1.frequency else (q0, q1)


def sinusoid(x, gate_repetition, amplitude, offset, phase):
    """Sinusoidal fit function."""
    return np.cos(gate_repetition * (x + phase)) * amplitude + offset


def fit_flux_amplitude(matrix, amps, times):
    """Estimate amplitude for CZ gate.


    Given the pattern of a chevron plot (see for example Fig. 2 here
    https://arxiv.org/pdf/1907.04818.pdf). This function estimates
    the CZ amplitude by finding the amplitude which gives the standard
    deviation, indicating that there are oscillation along the z axis.

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
    fs = []
    std = []
    for i in range(size_amp):
        y = matrix[i, :]
        period = fallback_period(guess_period(times, y))
        fs.append(1 / period)
        std.append(np.std(y))

    amplitude = amps[np.argmax(std)]
    delta = fs[np.argmax(std)]
    index = int(np.where(np.unique(amps) == amplitude)[0])
    return amplitude, index, delta


def phase_diff(phase_1, phase_2):
    """Return the phase difference of two sinusoids, normalized in the range [0, 2*pi]."""
    return np.mod(phase_2 - phase_1, 2 * np.pi)


def fit_sinusoid(thetas, data, gate_repetition):
    """Fit sinusoid to the given data."""
    pguess = [
        np.max(data) - np.min(data),
        np.mean(data),
        np.pi,
    ]

    popt, _ = curve_fit(
        lambda x, amplitude, offset, phase: sinusoid(
            x, gate_repetition, amplitude, offset, phase
        ),
        thetas,
        data,
        p0=pguess,
        bounds=(
            (0, -np.max(data), 0),
            (np.max(data), np.max(data), 2 * np.pi),
        ),
    )
    return popt.tolist()


def fit_virtualz(data: dict, pair: list, thetas: list, gate_repetition: int, key=None):
    fitted_parameters = {}
    angle = {}
    virtual_phase = {}
    leakage = {}
    fitted_param = {}
    if key is None:
        key = pair

    target, control = pair

    for setup in ["I", "X"]:
        target_data = data[target, control, setup].target
        try:
            params = fit_sinusoid(np.array(thetas), target_data, gate_repetition)
            fitted_param[target, control, setup] = params
        except Exception as e:
            log.warning(f"CZ fit failed for pair ({target, control}) due to {e}.")

    for setup in ["I", "X"]:
        # leakage estimate: L = m /2
        # See NZ paper from Di Carlo
        # approximation which does not need qutrits
        # https://arxiv.org/pdf/1903.02492.pdf
        leakage[key] = 0.5 * float(
            np.mean(
                data[target, control, "X"].control - data[target, control, "I"].control
            )
        )

        angle[key] = phase_diff(
            fitted_param[target, control, "X"][2],
            fitted_param[target, control, "I"][2],
        )
        virtual_phase[key] = fitted_param[target, control, "I"][2]
        fitted_parameters[key, setup] = fitted_param[target, control, setup]

    return fitted_parameters, virtual_phase, angle, leakage
