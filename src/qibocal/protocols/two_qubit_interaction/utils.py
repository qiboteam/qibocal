import numpy as np
import sympy
from qibolab import Platform
from scipy.optimize import curve_fit

from qibocal.auto.operation import QubitId, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ..utils import fallback_period, guess_period


def order_pair(pair: QubitPairId, platform: Platform) -> tuple[QubitId, QubitId]:
    """Order a pair of qubits by drive frequency."""
    q0, q1 = pair

    drive0 = platform.config(platform.qubits[q0].drive)
    drive1 = platform.config(platform.qubits[q1].drive)
    return (q1, q0) if drive0.frequency > drive1.frequency else (q0, q1)


def determine_cz_hybridization_state(
    calibration_platform: CalibrationPlatform, pair: QubitPairId
) -> tuple[dict[int, QubitId], list]:
    """Determine which qubit occupies each level in the CZ hybridized states.

    Uses calibrated frequencies, anharmonicities, and flux coefficients to
    find the flux roots of the 11–02 and 11–20 resonances. The nearest roots
    to zero are compared to infer which qubit occupies levels 0 and 2.
    This function returns a tuple containing the mapping from level (0 or 2) to qubit
    identifier and the flux values where resonance occurs.
    """

    assert calibration_platform.calibration is not None, (
        "Calibration data not available."
    )
    assert pair in calibration_platform.calibration.two_qubits, (
        "Qubit pair not found in the two-qubit system."
    )

    low_qubit, high_qubit = order_pair(pair, calibration_platform)

    x = sympy.symbols("x")

    high_qubit_flux = calibration_platform.calibration.single_qubits[
        high_qubit
    ].qubit.flux_coefficients
    high_qubit_f01 = calibration_platform.calibration.single_qubits[
        high_qubit
    ].qubit.frequency_01
    high_qubit_eta = calibration_platform.calibration.single_qubits[
        high_qubit
    ].qubit.anharmonicity

    assert high_qubit_flux, "qubits must be flux tunable."

    low_qubit_f01 = calibration_platform.calibration.single_qubits[
        low_qubit
    ].qubit.frequency_01
    low_qubit_eta = calibration_platform.calibration.single_qubits[
        low_qubit
    ].qubit.anharmonicity

    assert low_qubit_f01 and high_qubit_f01, "qubits must have frequency 01 defined."
    assert low_qubit_eta and high_qubit_eta, "qubits must have anharmonicity defined."

    # in this convention the first qubit is always the low frequency qubit
    # flux dependence for the high frequncy qubit
    high_qubit_func = (
        x**2 * high_qubit_flux[0]
        + x * high_qubit_flux[1]
        + high_qubit_flux[2]
        + high_qubit_f01 * 1e-9
    )
    # flux dependence for 11 state
    state_11 = high_qubit_func + low_qubit_f01 * 1e-9
    # flux dependence of the 20 state
    state_20 = (2 * low_qubit_f01 + low_qubit_eta) * 1e-9
    # flux dependence for 02 state
    state_02 = 2 * high_qubit_func + high_qubit_eta * 1e-9

    flux_resonance_11_02 = sympy.solve(sympy.Eq(state_11, state_02), x)
    flux_resonance_11_20 = sympy.solve(sympy.Eq(state_11, state_20), x)

    # If Omega(high)_12 < Omega(low)_01, the 11 and 02 levels never cross
    # as flux is varied. In that case there is no real 11-02 resonance point
    # to compare with the 11-20 crossing, which always occurs.
    # In fact, since Omega(20) = 2 * Omega(low)_01 + anharm(low)
    # and Omega(11) = Omega(low)_01 + Omega(high)_01, but Omega(high)_12 < Omega(low)_01 so for
    # transmons (negative anharmonicity), then is also true that
    # Omega(high)_01 > Omega(low)_01 + anharm(low).
    if all(fx.is_real for fx in flux_resonance_11_02):
        smallest_11_02 = min(flux_resonance_11_02, key=abs)
        smallest_11_20 = min(flux_resonance_11_20, key=abs)

        if smallest_11_02 <= smallest_11_20:
            return {0: low_qubit, 2: high_qubit}, flux_resonance_11_02

    return {0: high_qubit, 2: low_qubit}, flux_resonance_11_20


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
):
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
            log.warning(f"CZ fit failed for pair ({target, control}) due to {e}.")

    for setup in ["I", "X"]:
        # leakage estimate: L = m /2
        # See NZ paper from Di Carlo
        # approximation which does not need qutrits
        # https://arxiv.org/pdf/1903.02492.pdf
        leakage[key] = 0.5 * float(
            np.mean(data[target, control, "X"][1] - data[target, control, "I"][1])
        )

        angle[key] = phase_diff(
            fitted_param[target, control, "X"][2],
            fitted_param[target, control, "I"][2],
        )
        virtual_phase[key] = fitted_param[target, control, "I"][2]
        fitted_parameters[key, setup] = fitted_param[target, control, setup]

    return fitted_parameters, virtual_phase, angle, leakage
