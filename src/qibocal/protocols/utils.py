from colorsys import hls_to_rgb
from enum import Enum
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from qibolab._core.components import Config
from scipy import constants
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.fitting.classifier import run

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
KB = constants.k
HBAR = constants.hbar
"""Chi2 output when errors list contains zero elements"""
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


def effective_qubit_temperature(
    prob_0: NDArray, prob_1: NDArray, qubit_frequency: float, nshots: int
):
    """Calculates the qubit effective temperature.

    The formula used is the following one:

    kB Teff = - hbar qubit_freq / ln(prob_1/prob_0)

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
        temp = -HBAR * qubit_frequency / (np.log(prob_1 / prob_0) * KB)
        dT_dp0 = temp / prob_0 / np.log(prob_1 / prob_0)
        dT_dp1 = -temp / prob_1 / np.log(prob_1 / prob_0)
        error = np.sqrt((dT_dp0 * error_prob_0) ** 2 + (dT_dp1 * error_prob_1) ** 2)
    except ZeroDivisionError:
        temp = np.nan
        error = np.nan
    return temp, error


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


def plot_results(data: Data, qubit: QubitId, qubit_states: list, fit: Results):
    """
    Plots for the qubit and qutrit classification.

    Args:
        data (Data): acquisition data
        qubit (QubitID): qubit
        qubit_states (list): list of qubit states available.
        fit (Results): fit results
    """
    figures = []
    models_name = data.classifiers_list
    qubit_data = data.data[qubit]
    grid = evaluate_grid(qubit_data)

    fig = make_subplots(
        rows=1,
        cols=len(models_name),
        horizontal_spacing=SPACING * 3 / len(models_name) * 3,
        vertical_spacing=SPACING,
        subplot_titles=[run.pretty_name(model) for model in models_name],
        column_width=[COLUMNWIDTH] * len(models_name),
    )

    for i, model in enumerate(models_name):
        if fit is not None:
            predictions = fit.grid_preds[qubit][i]
            fig.add_trace(
                go.Contour(
                    x=grid[:, 0],
                    y=grid[:, 1],
                    z=np.array(predictions).flatten(),
                    showscale=False,
                    colorscale=[get_color_state0(i), get_color_state1(i)],
                    opacity=0.2,
                    name="Score",
                    hoverinfo="skip",
                    showlegend=True,
                ),
                row=1,
                col=i + 1,
            )

        model = run.pretty_name(model)
        max_x = max(grid[:, 0])
        max_y = max(grid[:, 1])
        min_x = min(grid[:, 0])
        min_y = min(grid[:, 1])

        for state in range(qubit_states):
            state_data = qubit_data[qubit_data["state"] == state]

            fig.add_trace(
                go.Scatter(
                    x=state_data["i"],
                    y=state_data["q"],
                    name=f"{model}: state {state}",
                    legendgroup=f"{model}: state {state}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3),
                ),
                row=1,
                col=i + 1,
            )

            fig.add_trace(
                go.Scatter(
                    x=[np.average(state_data["i"])],
                    y=[np.average(state_data["q"])],
                    name=f"{model}: state {state}",
                    legendgroup=f"{model}: state {state}",
                    showlegend=False,
                    mode="markers",
                    marker=dict(size=10),
                ),
                row=1,
                col=i + 1,
            )

        fig.update_xaxes(
            title_text="i [a.u.]",
            range=[min_x, max_x],
            row=1,
            col=i + 1,
            autorange=False,
            rangeslider=dict(visible=False),
        )
        fig.update_yaxes(
            title_text="q [a.u.]",
            range=[min_y, max_y],
            scaleanchor="x",
            scaleratio=1,
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        autosize=False,
        height=COLUMNWIDTH,
        width=COLUMNWIDTH * len(models_name),
        title=dict(text="Results", font=dict(size=TITLE_SIZE)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="left",
            y=-0.3,
            x=0,
            itemsizing="constant",
            font=dict(size=LEGEND_FONT_SIZE),
        ),
    )
    figures.append(fig)

    if fit is not None and len(models_name) != 1:
        fig_benchmarks = make_subplots(
            rows=1,
            cols=3,
            horizontal_spacing=SPACING,
            vertical_spacing=SPACING,
            subplot_titles=(
                "accuracy",
                "testing time [s]",
                "training time [s]",
            ),
        )
        for i, model in enumerate(models_name):
            for plot in range(3):
                fig_benchmarks.add_trace(
                    go.Scatter(
                        x=[model],
                        y=[fit.benchmark_table[qubit][i][plot]],
                        mode="markers",
                        showlegend=False,
                        marker=dict(size=10, color=get_color_state1(i)),
                    ),
                    row=1,
                    col=plot + 1,
                )

        fig_benchmarks.update_yaxes(type="log", row=1, col=2)
        fig_benchmarks.update_yaxes(type="log", row=1, col=3)
        fig_benchmarks.update_layout(
            autosize=False,
            height=COLUMNWIDTH,
            width=COLUMNWIDTH * 3,
            title=dict(text="Benchmarks", font=dict(size=TITLE_SIZE)),
        )

        figures.append(fig_benchmarks)
    return figures


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


def angle_wrap(angle: float):
    """Wrap an angle from [-np.inf,np.inf] into the [0,2*np.pi] domain"""
    return angle % (2 * np.pi)


def guess_period(x, y):
    """Return fft period estimation given a sinusoidal plot."""
    fft = np.fft.rfft(y)
    fft_freqs = np.fft.rfftfreq(len(y), d=(x[1] - x[0]))
    mags = np.abs(fft)
    mags[0] = 0
    return 1 / fft_freqs[np.argmax(mags)]


def guess_frequency_numpyfied(x: np.ndarray, y: np.ndarray, axis: int = -1):
    """Numpyfied version of :func:`guess_period` but here we work on frequencies."""
    assert x.ndim == 1, f"Expected 1D array, got array with shape {x.shape}"

    fft = np.fft.rfft(y, axis=axis)
    fft_freqs = np.fft.rfftfreq(y.shape[axis], d=(x[1] - x[0]))
    mags = np.abs(fft)
    mags[0] = 0

    selected_freqs = fft_freqs[np.argmax(mags, axis=axis)]

    return selected_freqs


def fallback_period(period):
    """Function to estimate period if guess_period fails."""
    return period if period is not None else 4


def fallback_frequency_numpyfied(frequency: np.ndarray):
    """Numpyfied version of :func:`fallback_period`, but here we work on frequencies."""
    assert frequency.ndim <= 1, (
        f"Expected 1D array or scalar, got array with shape {frequency.shape}"
    )

    return np.where(np.isnan(frequency), 4, frequency)


def quinn_fernandes_algorithm(
    signal_id: Any,
    x: Any,
    fs: float,
    speedup_flag: bool = False,
    axis: int = -1,
    iterations: int = 100,
    tol: int = 1e-6,
) -> np.ndarray:
    """This is a custom implementation of the Quinn-Fernandes algorithm.

    The Quinn–Fernandes method is a high-accuracy frequency estimator based on
    phase interpolation of the discrete Fourier transform (DFT). It refines the
    peak frequency obtained from the FFT by analyzing the phase evolution of the
    complex spectrum, achieving super-resolution beyond the FFT bin spacing.
    If :const:`speedup_flag` is set to `True`, the algorithm will change the updating rule,
    can lead to faster convergence, especially when the initial guess is close to the true frequency.
    Link for the original paper: https://www.jstor.org/stable/2337018?seq=3
    """

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if not isinstance(signal_id, np.ndarray):
        signal_id = np.array(signal_id)

    omegas = (
        2
        * np.pi
        * fallback_frequency_numpyfied(
            guess_frequency_numpyfied(x, signal_id, axis=axis)
        )
    )
    alpha = 2 * np.cos(omegas)

    signal_id = signal_id - np.mean(signal_id, axis=axis, keepdims=True)

    sig_shape = list(signal_id.shape)
    sig_shape[axis] += 2
    buffer_beta = []
    for _ in range(iterations):
        xi = np.zeros(sig_shape)
        for t in range(2, xi.shape[axis]):
            xi[..., t] = signal_id[..., t - 2] + alpha * xi[..., t - 1] - xi[..., t - 2]

        beta = np.sum((xi[..., 2:] + xi[..., :-2]) * xi[..., 1:-1], axis=axis) / np.sum(
            xi[..., :-1] ** 2, axis=axis
        )
        if len(buffer_beta) >= 5:
            buffer_beta.pop(0)
        buffer_beta.append(beta)
        if np.all(np.abs(np.mean(buffer_beta, axis=0) - alpha) < tol):
            alpha = beta
            break

        if speedup_flag:
            alpha = beta
        else:
            alpha = 2 * beta - alpha

    alpha = np.clip(alpha, -2, 2)
    omega_est = np.arccos(alpha / 2)
    med_omega = np.median(omega_est)

    return med_omega * fs
