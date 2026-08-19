"""Physics-related helper functions for qibocal protocols."""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import curve_fit

from qibocal.auto.operation import QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from .constants import KB, H, PowerLevel


def readout_frequency(
    target: QubitId,
    platform: CalibrationPlatform,
    power_level: PowerLevel = PowerLevel.low,
    state=0,
) -> float:
    """Returns readout frequency depending on power level."""
    platform_frequency = platform.config(platform.qubits[target].probe).frequency
    bare_frequency = platform.calibration.single_qubits[target].resonator.bare_frequency
    dressed_frequency = platform.calibration.single_qubits[
        target
    ].resonator.dressed_frequency
    if state == 1:
        try:
            state_frequency = platform.calibration.single_qubits[
                target
            ].readout.qudits_frequency[state]
            if state_frequency is not None:
                return state_frequency
        except KeyError:
            pass
    if power_level is PowerLevel.high and bare_frequency is not None:
        return bare_frequency
    if dressed_frequency is not None:
        return dressed_frequency
    return platform_frequency


def lorentzian(frequency, amplitude, center, sigma):
    # http://openafox.com/science/peak-function-derivations.html
    return (amplitude / np.pi) * (sigma / ((frequency - center) ** 2 + sigma**2))


def lorentzian_with_linear_background(
    frequency, amplitude, center, sigma, offset, slope
):
    peak = lorentzian(frequency, amplitude, center, sigma)
    background = offset + frequency * slope
    return peak + background


def lorentzian_fit(data, resonator_type=None, fit=None):
    frequencies = data.freq
    signal = data.signal

    # Guess parameters for Lorentzian max or min
    guess_slope = (signal[-1] - signal[0]) / (frequencies[-1] - frequencies[0])
    guess_offset = signal[0] - guess_slope * frequencies[0]
    guess_background = guess_offset + guess_slope * frequencies
    voltages_no_background = signal - guess_background

    if (resonator_type == "3D" and fit == "resonator") or (
        resonator_type == "2D" and fit == "qubit"
    ):
        guess_center = frequencies[np.argmax(voltages_no_background)]
        guess_peak_height = voltages_no_background.max()
        indices_beyond_half = np.where(voltages_no_background > guess_peak_height / 2)[
            0
        ]
    else:
        guess_center = frequencies[np.argmin(voltages_no_background)]
        guess_peak_height = voltages_no_background.min()
        indices_beyond_half = np.where(voltages_no_background < guess_peak_height / 2)[
            0
        ]

    if len(indices_beyond_half) >= 1:
        guess_sigma = (
            frequencies[indices_beyond_half[-1]] - frequencies[indices_beyond_half[0]]
        ) / 2
    else:
        # if there is no clear peak, we give a high flexibility
        guess_sigma = frequencies[-1] - frequencies[0]

    guess_amp = guess_peak_height * guess_sigma * np.pi

    initial_parameters = [
        guess_amp,
        guess_center,
        guess_sigma,
        guess_offset,
        guess_slope,
    ]
    freq_domain_size = frequencies[-1] - frequencies[0]
    bounds = (
        [-np.inf, frequencies[0], 0.0, -np.inf, -np.inf],
        [np.inf, frequencies[-1], freq_domain_size, np.inf, np.inf],
    )

    # fit the model with the data and guessed parameters
    try:
        fit_parameters, parameters_cov = curve_fit(
            lorentzian_with_linear_background,
            frequencies,
            signal,
            p0=initial_parameters,
            bounds=bounds,
        )
        # The output results are stored in a json, but ndarray is not JSON serializable,
        # so the parameters are converted to list.
        parameter_errors = np.sqrt(np.diag(parameters_cov)).tolist()
        model_parameters = fit_parameters.tolist()
        return model_parameters[1], model_parameters, parameter_errors
    except RuntimeError as e:
        log.warning(f"Lorentzian fit not successful due to {e}")


def effective_qubit_temperature(
    prob_0: NDArray, prob_1: NDArray, qubit_frequency: float, nshots: int
):
    """Calculates the qubit effective temperature.

    The formula used is the following one:

    kB Teff = - h qubit_freq / ln(prob_1/prob_0)

    Args:
        prob_0 (NDArray): population 0 samples
        prob_1 (NDArray): population 1 samples
        qubit_frequency(float): frequency of qubit
        nshots (int): number of shots
    Returns:
        temp (float): effective temperature
        error (float): error on effective temperature

    """
    error_prob_0 = np.sqrt(prob_0 * (1 - prob_0) / nshots)
    error_prob_1 = np.sqrt(prob_1 * (1 - prob_1) / nshots)
    # TODO: find way to handle this exception
    try:
        temp = -H * qubit_frequency / (np.log(prob_1 / prob_0) * KB)
        dT_dp0 = temp / prob_0 / np.log(prob_1 / prob_0)
        dT_dp1 = -temp / prob_1 / np.log(prob_1 / prob_0)
        error = np.sqrt((dT_dp0 * error_prob_0) ** 2 + (dT_dp1 * error_prob_1) ** 2)
    except ZeroDivisionError:
        temp = np.nan
        error = np.nan
    return temp, error


def euclidean_metric(point1: np.ndarray, point2: np.ndarray):
    """Euclidean distance between two arrays."""
    return np.linalg.norm(point1 - point2)


def angle_wrap(angle: float):
    """Wrap an angle from [-np.inf,np.inf] into the [0,2*np.pi] domain"""
    return angle % (2 * np.pi)


def baseline_als(data: NDArray, lamda: float, p: float, niter: int = 10) -> NDArray:
    """Estimate data baseline with "asymmetric least squares" method.

    The :obj:`lambda` parameter controls the stiffness weight. A larger value will
    suppress more and more the fluctuations in the estimated baseline.
    The :obj:`p` parameters controls instead the asymmetry, deweighting fluctuations in
    one direction only.

    The convergence is iterative, but it is often sufficiently fast that a closed loop
    with a predetermined number of iterations is enough. :obj:`niter` allows changing
    the amount of iterations.

    The approach is defined in

    Eilers, Paul & Boelens, Hans. (2005). Baseline Correction with Asymmetric Least
    Squares Smoothing. Unpubl. Manuscr.

    """
    n_obs = len(data)
    diff = sparse.csr_array(np.diff(np.eye(n_obs), 2))
    weights = np.ones(n_obs)
    for _ in range(niter):
        weights_mat = sparse.diags_array(weights)
        a = weights_mat + lamda * diff.dot(diff.transpose())
        b = weights * data
        z = sparse.linalg.spsolve(a, b)
        weights = p * (data > z) + (1 - p) * (data < z)
    return z
