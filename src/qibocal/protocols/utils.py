from colorsys import hls_to_rgb
from enum import Enum
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from qibolab._core.components import Config
from scipy import constants, sparse
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from qibocal.auto.operation import Data, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

GHZ_TO_HZ = 1e9
HZ_TO_GHZ = 1e-9
V_TO_UV = 1e6
S_TO_NS = 1e9
MESH_SIZE = 50
MARGIN = 0
SPACING = 0.1
COLUMNWIDTH = 600
LEGEND_FONT_SIZE = 20
TITLE_SIZE = 25
EXTREME_CHI = 1e4
"""Chi2 output when errors list contains zero elements"""
KB = constants.k
H = constants.h
COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"
CONFIDENCE_INTERVAL_FIRST_MASK = 99
"""Confidence interval used to mask flux data."""
CONFIDENCE_INTERVAL_SECOND_MASK = 70
"""Confidence interval used to clean outliers."""
DELAY_FIT_PERCENTAGE = 10
"""Percentage of the first and last points used to fit the cable delay."""
STRING_TYPE = "<U100"


class PowerLevel(str, Enum):
    """Power Regime for Resonator Spectroscopy"""

    high = "high"
    low = "low"


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
    if power_level is PowerLevel.high:
        if bare_frequency is not None:
            return bare_frequency
    if dressed_frequency is not None:
        return dressed_frequency
    return platform_frequency


def lorentzian(frequency, amplitude, center, sigma, offset):
    # http://openafox.com/science/peak-function-derivations.html
    return (amplitude / np.pi) * (
        sigma / ((frequency - center) ** 2 + sigma**2)
    ) + offset


def lorentzian_fit(data, resonator_type=None, fit=None):
    frequencies = data.freq * HZ_TO_GHZ
    voltages = data.signal

    # Guess parameters for Lorentzian max or min
    # TODO: probably this is not working on HW
    guess_offset = np.mean(
        voltages[np.abs(voltages - np.mean(voltages)) < np.std(voltages)]
    )
    if (resonator_type == "3D" and fit == "resonator") or (
        resonator_type == "2D" and fit == "qubit"
    ):
        guess_center = frequencies[
            np.argmax(voltages)
        ]  # Argmax = Returns the indices of the maximum values along an axis.
        guess_sigma = abs(frequencies[np.argmin(voltages)] - guess_center)
        guess_amp = (np.max(voltages) - guess_offset) * guess_sigma * np.pi

    else:
        guess_center = frequencies[
            np.argmin(voltages)
        ]  # Argmin = Returns the indices of the minimum values along an axis.
        guess_sigma = abs(frequencies[np.argmax(voltages)] - guess_center)
        guess_amp = (np.min(voltages) - guess_offset) * guess_sigma * np.pi

    initial_parameters = [
        guess_amp,
        guess_center,
        guess_sigma,
        guess_offset,
    ]
    # fit the model with the data and guessed parameters
    try:
        if hasattr(data, "error_signal"):
            if not np.isnan(data.error_signal).any():
                fit_parameters, perr = curve_fit(
                    lorentzian,
                    frequencies,
                    voltages,
                    p0=initial_parameters,
                    sigma=data.error_signal,
                )
                perr = np.sqrt(np.diag(perr)).tolist()
                model_parameters = list(fit_parameters)
                return model_parameters[1] * GHZ_TO_HZ, list(model_parameters), perr
        fit_parameters, perr = curve_fit(
            lorentzian,
            frequencies,
            voltages,
            p0=initial_parameters,
        )
        perr = [0] * 4
        model_parameters = list(fit_parameters)
        return model_parameters[1] * GHZ_TO_HZ, model_parameters, perr
    except RuntimeError as e:
        log.warning(f"Lorentzian fit not successful due to {e}")


class DcFilteredConfig(Config):
    """Dummy config for dc with filters.

    Required by cryoscope protocol.

    """

    kind: Literal["dc-filter"] = "dc-filter"
    offset: float
    filter: list


def effective_qubit_temperature(predictions, qubit_frequency: float, nshots: int):
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
    prob_1 = np.count_nonzero(predictions) / len(predictions)
    prob_0 = 1 - prob_1
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


def compute_qnd(
    ones_first_measure,
    zeros_first_measure,
    ones_second_measure,
    zeros_second_measure,
    pi=False,
) -> tuple[float, list, list]:
    """QND calculation.

    For the standard QND we follow https://arxiv.org/pdf/2106.06173
    for the pi variant we follow https://arxiv.org/pdf/2110.04285

    Returns the QND and the two measurement matrices."""

    p_m1 = np.mean([zeros_first_measure, ones_first_measure], axis=1)
    p_m2 = np.mean([zeros_second_measure, ones_second_measure], axis=1)

    lambda_m = np.stack([1 - p_m1, p_m1])
    lambda_m2 = np.stack([1 - p_m2, p_m2])

    # pinv to avoid tests failing due to singular matrix
    p_o = np.linalg.pinv(lambda_m) @ lambda_m2

    qnd = np.sum(np.diag(p_o)) / 2 if not pi else np.sum(np.diag(p_o[::-1])) / 2
    return qnd, lambda_m.tolist(), lambda_m2.tolist()


def compute_assignment_fidelity(
    one_samples: np.ndarray, zero_samples: np.ndarray
) -> float:
    """Computing assignment fidelity from shots.
    The first argument are the samples when preparing state 1 and the second argument are
    the samples when preparing state 0.
    """

    p_m1_i0 = np.mean(zero_samples)
    p_m1_i1 = np.mean(one_samples)
    p_m0_i1 = 1 - p_m1_i1

    # compute assignment fidelity
    fidelity = 1 - (p_m1_i0 + p_m0_i1) / 2
    return fidelity


def classify(arr: np.ndarray, angle: float, threshold: float) -> np.ndarray:
    """Mapping IQ array in 0s and 1s given angle and threshold."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    rotated = arr @ rot.T
    return (rotated[:, 0] > threshold).astype(int)


def norm(x_mags):
    return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))


def cumulative(input_data, points):
    r"""Evaluates in data the cumulative distribution
    function of `points`.
    """
    return np.searchsorted(np.sort(points), np.sort(input_data))


def fit_punchout(data: Data, fit_type: str):
    """
    Punchout fitting function.

    Args:

    data (Data): Punchout acquisition data.
    fit_type (str): Punchout type, it could be `amp` (amplitude)
    or `att` (attenuation).

    Return:

    List of dictionaries containing the low, high amplitude
    (attenuation) frequencies and the readout amplitude (attenuation)
    for each qubit.
    """
    qubits = data.qubits

    low_freqs = {}
    high_freqs = {}
    ro_values = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        freqs = qubit_data.freq
        amps = getattr(qubit_data, fit_type)
        signal = qubit_data.signal
        if data.resonator_type == "3D":
            mask_freq, mask_amps = extract_feature(
                freqs, amps, signal, "max", ci_first_mask=90
            )
        else:
            mask_freq, mask_amps = extract_feature(
                freqs, amps, signal, "min", ci_first_mask=90
            )
        if fit_type == "amp":
            best_freq = np.max(mask_freq)
            bare_freq = np.min(mask_freq)
        else:
            best_freq = np.min(mask_freq)
            bare_freq = np.max(mask_freq)
        ro_val = np.max(mask_amps[mask_freq == best_freq])
        low_freqs[qubit] = best_freq
        high_freqs[qubit] = bare_freq
        ro_values[qubit] = ro_val
    return [low_freqs, high_freqs, ro_values]


def eval_magnitude(value):
    """number of non decimal digits in `value`"""
    if value == 0 or not np.isfinite(value):
        return 0
    return int(np.floor(np.log10(abs(value))))


def round_report(
    measure: list,
) -> tuple[list, list]:
    """
    Rounds the measured values and their errors according to their significant digits.

    Args:
        measure (list): Variable-Errors couples.

    Returns:
        A tuple with the lists of values and errors in the correct string format.
    """
    rounded_values = []
    rounded_errors = []
    for value, error in measure:
        if value:
            magnitude = eval_magnitude(value)
        else:
            magnitude = 0

        ndigits = max(significant_digit(error * 10 ** (-1 * magnitude)), 0)
        if magnitude != 0:
            rounded_values.append(
                f"{round(value * 10 ** (-1 * magnitude), ndigits)}e{magnitude}"
            )
            rounded_errors.append(
                f"{np.format_float_positional(round(error * 10 ** (-1 * magnitude), ndigits), trim='-')}e{magnitude}"
            )
        else:
            rounded_values.append(f"{round(value * 10 ** (-1 * magnitude), ndigits)}")
            rounded_errors.append(
                f"{np.format_float_positional(round(error * 10 ** (-1 * magnitude), ndigits), trim='-')}"
            )

    return rounded_values, rounded_errors


def format_error_single_cell(measure: tuple):
    """Helper function to print mean value and error in one line."""
    # extract mean value and error
    mean = measure[0][0]
    error = measure[1][0]
    if all("e" in number for number in measure[0] + measure[1]):
        magn = mean.split("e")[1]
        return f"({mean.split('e')[0]} ± {error.split('e')[0]}) 10<sup>{magn}</sup>"
    return f"{mean} ± {error}"


def chi2_reduced(
    observed: NDArray,
    estimated: NDArray,
    errors: NDArray,
    dof: Optional[float] = None,
):
    if np.count_nonzero(errors) < len(errors):
        return EXTREME_CHI

    if dof is None:
        dof = len(observed) - 1

    chi2 = np.sum(np.square((observed - estimated) / errors)) / dof

    return chi2


def chi2_reduced_complex(
    observed: tuple[NDArray, NDArray],
    estimated: NDArray,
    errors: tuple[NDArray, NDArray],
    dof: Optional[float] = None,
):
    observed_complex = np.abs(observed[0] * np.exp(1j * observed[1]))

    observed_real = np.real(observed_complex)
    observed_imag = np.imag(observed_complex)

    estimated_real = np.real(estimated)
    estimated_imag = np.imag(estimated)

    observed_error_real = np.sqrt(
        (np.cos(observed[1]) * errors[0]) ** 2
        + (observed[0] * np.sin(observed[1]) * errors[1]) ** 2
    )
    observed_error_imag = np.sqrt(
        (np.sin(observed[1]) * errors[0]) ** 2
        + (observed[0] * np.cos(observed[1]) * errors[1]) ** 2
    )

    chi2_real = chi2_reduced(observed_real, estimated_real, observed_error_real, dof)
    chi2_imag = chi2_reduced(observed_imag, estimated_imag, observed_error_imag, dof)

    return chi2_real + chi2_imag


def get_color_state0(number):
    return "rgb" + str(hls_to_rgb((-0.35 - number * 9 / 20) % 1, 0.6, 0.75))


def get_color_state1(number):
    return "rgb" + str(hls_to_rgb((-0.02 - number * 9 / 20) % 1, 0.6, 0.75))


def significant_digit(number: float):
    """Computes the position of the first significant digit of a given number.

    Args:
        number (Number): number for which the significant digit is computed. Can be complex.

    Returns:
        int: position of the first significant digit. Returns ``-1`` if the given number
            is ``>= 1``, ``= 0`` or ``inf``.
    """

    if (
        np.isinf(np.real(number))
        or np.real(number) >= 1
        or number == 0
        or np.isnan(number)
    ):
        return -1

    position = max(np.ceil(-np.log10(abs(np.real(number)))), -1)

    if np.imag(number) != 0:
        position = max(position, np.ceil(-np.log10(abs(np.imag(number)))))

    return int(position)


def evaluate_grid(
    data: NDArray,
):
    """
    This function returns a matrix grid evaluated from
    the datapoints `data`.
    """
    max_x = (
        max(
            0,
            data["i"].max(),
        )
        + MARGIN
    )
    max_y = (
        max(
            0,
            data["q"].max(),
        )
        + MARGIN
    )
    min_x = (
        min(
            0,
            data["i"].min(),
        )
        - MARGIN
    )
    min_y = (
        min(
            0,
            data["q"].min(),
        )
        - MARGIN
    )
    i_values, q_values = np.meshgrid(
        np.linspace(min_x, max_x, num=MESH_SIZE),
        np.linspace(min_y, max_y, num=MESH_SIZE),
    )
    return np.vstack([i_values.ravel(), q_values.ravel()]).T


def table_dict(
    qubit: Union[list[QubitId], QubitId],
    names: list[str],
    values: list,
    display_error=False,
) -> dict:
    """
    Build a dictionary to generate HTML table with `table_html`.

    Args:
        qubit (Union[list[QubitId], QubitId]): If qubit is a scalar value,
        the "Qubit" entries will have only this value repeated.
        names (list[str]): List of the names of the parameters.
        values (list): List of the values of the parameters.
        display_errors (bool): if `True`, it means that `values` is a list of value-error couples,
        so an `Errors` key will be displayed in the dictionary. The function will round the couples according to their significant digits. Default False.

    Return:
        A dictionary with keys `Qubit`, `Parameters`, `Values` (`Errors`).
    """
    if not np.isscalar(values):
        if np.isscalar(qubit):
            qubit = [qubit] * len(names)

        if display_error:
            rounded_values, rounded_errors = round_report(values)

            return {
                "Qubit": qubit,
                "Parameters": names,
                "Values": rounded_values,
                "Errors": rounded_errors,
            }
    else:  # If `values` is scalar also `qubit` should be a scalar
        qubit = [
            qubit
        ]  # In this way when the Dataframe is generated, an index is not required.
    return {"Qubit": qubit, "Parameters": names, "Values": values}


def table_html(data: dict) -> str:
    """This function converts a dictionary into an HTML table.

    Args:
        data (dict): the keys will be converted into table entries and the
        values will be the columns of the table.
        Values must be valid HTML strings.

    Return:
        str
    """
    return pd.DataFrame(data).to_html(
        classes="fitting-table", index=False, border=0, escape=False
    )


def extract_feature(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    feat: str,
    ci_first_mask: float = CONFIDENCE_INTERVAL_FIRST_MASK,
    ci_second_mask: float = CONFIDENCE_INTERVAL_SECOND_MASK,
):
    """Extract feature using confidence intervals.

    Given a dataset of the form (x, y, z) where a spike or a valley is expected,
    this function discriminate the points (x, y) with a signal, from the pure noise
    and return the first ones.

    A first mask is construct by looking at `ci_first_mask` confidence interval for each y bin.
    A second mask is applied by looking at `ci_second_mask` confidence interval to remove outliers.
    `feat` could be `min` or `max`, in the first case the function will look for valleys, otherwise
    for peaks.

    """

    masks = []
    for i in np.unique(y):
        signal_fixed_y = z[y == i]
        min, max = np.percentile(
            signal_fixed_y,
            [100 - ci_first_mask, ci_first_mask],
        )
        masks.append(signal_fixed_y < min if feat == "min" else signal_fixed_y > max)

    first_mask = np.vstack(masks).ravel()
    min, max = np.percentile(
        z[first_mask],
        [100 - ci_second_mask, ci_second_mask],
    )
    second_mask = z[first_mask] < min if feat == "min" else z[first_mask] > max
    return x[first_mask][second_mask], y[first_mask][second_mask]


def guess_period(x, y):
    """Return fft period estimation given a sinusoidal plot."""

    fft = np.fft.rfft(y)
    fft_freqs = np.fft.rfftfreq(len(y), d=(x[1] - x[0]))
    mags = abs(fft)
    mags[0] = 0
    local_maxima, _ = find_peaks(mags)
    if len(local_maxima) > 0:
        return 1 / fft_freqs[np.argmax(mags)]
    return None


def fallback_period(period):
    """Function to estimate period if guess_period fails."""
    return period if period is not None else 4


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
