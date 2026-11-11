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


def sinusoid(x, amplitude, offset, phase):
    """Sinusoidal fit function."""
    return np.cos(x + phase) * amplitude + offset


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
        period = fallback_period(guess_period(times, y)).tolist()
        fs.append(1 / period)
        std.append(np.std(y))

    amplitude = amps[np.argmax(std)]
    delta = fs[np.argmax(std)]
    index = int(np.where(np.unique(amps) == amplitude)[0][0])
    return amplitude, index, delta


def phase_diff(phase_1, phase_2):
    """Return the phase difference of two sinusoids, normalized in the range [0, 2*pi]."""
    return np.mod(phase_2 - phase_1, 2 * np.pi)


def best_snz_point(angles, leakages, x, y):
    """Return ``(x, y, angle, leakage)`` of the lowest-leakage grid point with an
    angle within 10% of pi, or ``None`` if there is no such point."""
    mask = (angles > np.pi * 0.9) & (angles < np.pi * 1.1)
    if not mask.any():
        return None
    i = leakages[mask].argmin()
    return (
        x[mask][i].item(),
        y[mask][i].item(),
        angles[mask][i].item(),
        leakages[mask][i].item(),
    )


def fit_sinusoid(thetas, data):
    """Fit sinusoid to the given data."""
    pguess = [
        np.max(data) - np.min(data),
        np.mean(data),
        np.pi,
    ]

    popt, _ = curve_fit(
        sinusoid,
        thetas,
        data,
        p0=pguess,
        bounds=(
            (0, -np.max(data), 0),
            (np.max(data), np.max(data), 2 * np.pi),
        ),
    )
    return popt.tolist()


def fit_virtualz(
    data: dict,
    pair: list,
    thetas: list,
    key=None,
    native: str = "CZ",
):
    r"""Extract virtual phase, angle and leakage from the virtual Z experiment.

    For CZ the phase of the ``I`` setup is the single-qubit phase acquired by the
    target, and the difference between the ``X`` and ``I`` phases is the conditional
    phase (ideally :math:`\pi`).

    For iSWAP the target phase is measured after the excitation is transferred from
    the partner qubit, :math:`|01\rangle \to i e^{i\phi_t}|10\rangle`, so the ``I``
    phase is :math:`\phi_t + \pi/2` and the virtual phase is :math:`\phi_t`. The
    ``X`` phase is :math:`\phi_{11} - \phi_c - \pi/2`, so the returned angle
    :math:`\phi_{11} - \phi_t - \phi_c` is the residual conditional phase, which is
    0 for an ideal iSWAP and cannot be corrected with virtual Z rotations.
    """
    fitted_parameters = {}
    angle = {}
    virtual_phase = {}
    leakage = {}
    fitted_param = {}
    if key is None:
        key = pair

    target, control = pair
    for setup in ["I", "X"]:
        target_data = data[target, control, setup][0]
        try:
            params = fit_sinusoid(np.array(thetas), target_data)
            fitted_param[target, control, setup] = params
        except Exception as e:
            log.warning(f"{native} fit failed for pair ({target, control}) due to {e}.")

    for setup in ["I", "X"]:
        # leakage estimate: L = m /2
        # See NZ paper from Di Carlo
        # approximation which does not need qutrits
        # https://arxiv.org/pdf/1903.02492.pdf
        leakage[key] = 0.5 * float(
            np.mean(data[target, control, "X"][1] - data[target, control, "I"][1])
        )

        phase_i = fitted_param[target, control, "I"][2]
        phase_x = fitted_param[target, control, "X"][2]
        if native == "iSWAP":
            angle[key] = float(np.mod(phase_x - phase_i + np.pi, 2 * np.pi))
            virtual_phase[key] = float(np.mod(phase_i - np.pi / 2, 2 * np.pi))
        else:
            angle[key] = phase_diff(phase_x, phase_i)
            virtual_phase[key] = phase_i
        fitted_parameters[key, setup] = fitted_param[target, control, setup]

    return fitted_parameters, virtual_phase, angle, leakage
