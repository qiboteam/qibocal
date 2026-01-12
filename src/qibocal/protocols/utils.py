from colorsys import hls_to_rgb
from enum import Enum
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from qibolab._core.components import Config
from scipy import constants, ndimage, sparse
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import norm as scipy_norm
from sklearn.cluster import HDBSCAN

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
"""Chi2 output when errors list contains zero elements"""
KB = constants.k
H = constants.h
COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"
DELAY_FIT_PERCENTAGE = 10
"""Percentage of the first and last points used to fit the cable delay."""
STRING_TYPE = "<U100"

MAX_PIXELS = 2
"""How many pixels at most two clusters' endpoints should be far for merging them."""
DISTANCE_XY = 1.5 * MAX_PIXELS  # very heuristic
""" Minimum distance for separate clusters.
Clusters below this distance will be merged.
Since it is given in a 3D-space, with a compressed vertical dimension, and the horizontal plane measured in pixels,
this distance correspond to diagonally adjacent pixels, with some additional leeway for the extra dimension.
"""
DISTANCE_Z = 0.5
"""See :const:`DISTANCE_XY`."""


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
        mask_freq, mask_amps = extract_feature(freqs, amps, signal, data.find_min)

        if mask_freq.size == 0:  # mask_freq and mask_amps have always the same shape
            best_freq = 0
            bare_freq = 0
            ro_val = 0
        else:
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
        rows=2,
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

        # Colorset for plots
        COLORS = px.colors.qualitative.Plotly[0:qubit_states]
        if COLORS[0].startswith("#"):
            COLORS = [
                f"rgba({int(COLORS[j][1:3], 16)},{int(COLORS[j][3:5], 16)},{int(COLORS[j][5:7], 16)},0.5)"
                for j in range(len(COLORS))
            ]

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
                    marker_color=COLORS[state],
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

            # Add 1D histogram trace rotated by rot_angle from the fit results
            if fit is not None and getattr(fit, "rotation_angle", None) is not None:
                rot_angle = np.round(fit.rotation_angle[qubit], 3)
                threshold = np.round(fit.threshold[qubit], 3)

                x, y = state_data["i"], state_data["q"]
                c, s = np.cos(rot_angle), np.sin(rot_angle)
                rot = np.array([[c, -s], [s, c]])
                rotated = np.vstack([x, y]).T @ rot.T
                rotated[:, 0] = rotated[:, 0]

                # histogram using only the x values
                hist, bin_edges = np.histogram(
                    rotated[:, 0],
                    bins=30,
                    density=True,
                )
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Gaussian fit to histogram
                mu, std = scipy_norm.fit(rotated[:, 0])
                pdf = scipy_norm.pdf(bin_centers, mu, std)

                fig.add_trace(
                    go.Bar(
                        x=bin_centers - threshold,
                        y=hist,
                        name=f"{model}: state {state} hist",
                        legendgroup=f"{model}: state {state}",
                        showlegend=False,
                        marker=dict(color=COLORS[state]),
                        width=(bin_centers[1] - bin_centers[0])
                        if len(bin_centers) > 1
                        else 0.1,
                    ),
                    row=2,
                    col=i + 1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=bin_centers - threshold,
                        y=pdf,
                        name=f"{model}: state {state} norm fit",
                        mode="lines",
                        legendgroup=f"{model}: state {state}",
                        showlegend=False,
                        line=dict(width=2, color=COLORS[state]),
                    ),
                    row=2,
                    col=i + 1,
                )

                # Add vertical line for threshold
                fig.add_trace(
                    go.Scatter(
                        x=[0, 0],
                        y=[0, max(hist) * 1.1],
                        name="threshold",  # No name for legend
                        mode="lines",
                        line=dict(color="black", width=2, dash="dot"),
                        showlegend=False,
                    ),
                    row=2,
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


def euclidean_metric(point1: np.ndarray, point2: np.ndarray):
    """Euclidean distance between two arrays."""
    return np.linalg.norm(point1 - point2)


def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    https://en.wikipedia.org/wiki/Whitening_transformation

    X: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """
    assert X.ndim == 2
    EPS = 10e-5

    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1.0 / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)

    X_white = np.dot(X, W)

    return X_white


def filter_data(matrix_z: np.ndarray):
    """Filter data with a ZCA transformation and then a unit-variance Gaussian."""

    # adding zca filter for filtering out background noise gradient
    zca_z = zca_whiten(matrix_z)
    # adding gaussian fliter with unitary variance for blurring the signal and reducing noise
    return ndimage.gaussian_filter(zca_z, 1)


def scaling_global(sig: np.ndarray) -> np.ndarray:
    """Min–max scaling over the whole np.ndarray (global)."""
    return scaling_slice(sig, axis=None)


def scaling_slice(sig: np.ndarray, axis: Optional[int]) -> np.ndarray:
    """Min–max scaling over a specific axis of the np.ndarray."""

    def expand(a):
        return np.expand_dims(a, axis) if axis is not None else a

    sig_min = expand(np.min(sig, axis=axis))
    return (sig - sig_min) / (expand(np.max(sig, axis=axis)) - sig_min)


def horizontal_diagonal(xs: np.ndarray, ys: np.ndarray) -> float:
    """Computing the lenght of the diagonal of a two dimensional grid."""
    sizes = np.empty(2)
    for i, values in enumerate([xs, ys]):
        sizes[i] = np.max(values) - np.min(values)
    return np.sqrt((sizes**2).sum())


def build_clustering_data(peaks_dict: dict, z: np.ndarray):
    """Preprocessing of the data to cluster."""
    x_ = peaks_dict["x"]["idx"]
    y_ = peaks_dict["y"]["idx"]
    z_ = z[y_, x_]

    return np.stack((x_, y_, scaling_global(z_))).T


def peaks_finder(x, y, z) -> dict:
    """Function for finding the peaks over the whole signal.

    This function takes as input 3 features of the signal.
    It slices the dataset along a preferred direction (`y` dimension, corresponding to the flux bias) and for each slice it determines the biggest peaks
    by using `scipy.signal.find_peaks` routine.
    It returns a dictionary `peaks_dict` containing all the features for the computed peaks.
    """

    # filter data using find_peaks
    peaks = {"x": {"idx": [], "val": []}, "y": {"idx": [], "val": []}}
    for y_idx, y_val in enumerate(y):
        peak, info = find_peaks(z[y_idx], prominence=0.2)
        if len(peak) > 0:
            idx = np.argmax(info["prominences"])
            # if multiple peaks per bias are found, select the one with the highest prominence
            x_idx = peak[idx]
            peaks["x"]["idx"].append(x_idx)
            peaks["x"]["val"].append(x[x_idx])
            peaks["y"]["idx"].append(y_idx)
            peaks["y"]["val"].append(y_val)

    return {
        feat: {kind: np.array(vals) for kind, vals in smth.items()}
        for feat, smth in peaks.items()
    }


def merging(
    data: tuple,
    labels: list,
    min_points_per_cluster: int,
    distance_xy: float,
    distance_z: float,
) -> list[bool]:
    """Divides the processed signal into clusters for separating signal from noise.

    `data` is a 3D tuple of the data to cluster, while `labels` is the classification made by the clustering algorithm;
    `min_points_per_cluster` is the minimum size of points for a cluster to be considered relevant signal.
    It is also possible to set the parameter `distance`, which represents the Euclidean distance between neighboring points of two clusters.
    If this distance is smaller than `distance`, the two clusters are merged.
    It allows a `min_cluster_size=2` in order to decrease as much as possible misclassification of few points.
    The function returns a boolean list corresponding to the indices of the relevant signal.
    """

    unique_labels = np.unique(labels)

    indices_list = np.arange(len(labels)).astype(int)
    indexed_labels = np.stack((labels, indices_list)).T
    data = np.vstack((data.T, indices_list))

    clusters = [data[:, labels == lab] for lab in unique_labels if lab >= 0]
    noise_points = data[:, labels < 0]

    for i in range(noise_points.shape[1]):
        clusters.append(noise_points[:, i][:, np.newaxis])

    clusters = sorted(
        clusters,
        key=lambda c: np.min(c[1]),
    )

    first = clusters[0]
    first_leftmost = first[:, np.argmin(first[1, :])]
    first_rightmost = first[:, np.argmax(first[1, :])]
    first_label = indexed_labels[first_leftmost[3].astype(int), 0]

    active_clusters = {
        first_label: {
            "cluster": first,
            "leftmost": first_leftmost,
            "rightmost": first_rightmost,
        }
    }

    for cluster in clusters[1:]:
        distances_list = []
        indices = []

        for idx in active_clusters.keys():
            cluster_rightmost = cluster[:, np.argmax(cluster[1, :])]
            cluster_leftmost = cluster[:, np.argmin(cluster[1, :])]
            cluster_label = indexed_labels[cluster_leftmost[3].astype(int), 0]

            d_xy = euclidean_metric(
                active_clusters[idx]["rightmost"][:-2], cluster_leftmost[:-2]
            )
            d_z = euclidean_metric(
                active_clusters[idx]["rightmost"][-2], cluster_leftmost[-2]
            )
            if d_xy <= distance_xy and d_z <= distance_z:  # keep the list
                distances_list.append(np.sqrt(d_xy**2 + d_z**2))
                indices.append(idx)

        if len(distances_list) != 0:
            best_dist = np.argmin(distances_list)
            best_idx = indices[best_dist]
            old_cluster = active_clusters[best_idx]["cluster"]
            updated_cluster = np.hstack((old_cluster, cluster))
            active_clusters[best_idx]["cluster"] = updated_cluster
            active_clusters[best_idx]["rightmost"] = updated_cluster[
                :, np.argmax(updated_cluster[1, :])
            ]
        else:
            if cluster_label < 0:
                cluster_label = np.max(unique_labels) + 1
                unique_labels = np.append(unique_labels, cluster_label)

            active_clusters[cluster_label] = {
                "cluster": cluster,
                "leftmost": cluster_leftmost,
                "rightmost": cluster_rightmost,
            }

    valid_clusters = {
        lab: v_clust
        for lab, v_clust in active_clusters.items()
        if v_clust["cluster"].shape[1] >= min_points_per_cluster
    }

    # since we allowed for clustering even a group of 2 points, we filter the allowed eligible clusters
    # to be at least composed by a minimum number of points given by min_points_per_cluster parameter

    medians = np.array(
        [[lab, np.median(cl["cluster"][2, :])] for lab, cl in valid_clusters.items()]
    )
    # we only take the first three values of each point in the cluster because they correspond to the 3 features (x,y,z)

    signal_labels = np.zeros(indices_list.size, dtype=bool)
    if len(medians) != 0:
        signal_idx = medians[np.argmax(medians[:, 1]), 0]
        signal_labels[valid_clusters[signal_idx]["cluster"][-1, :].astype(int)] = True

    return signal_labels


def extract_feature(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, find_min: bool, min_points: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features of the signal by filtering out background noise.

    It first applies a custom filter mask (see `custom_filter_mask`)
    and then finds the biggest peak for each DC bias value;
    the masked signal is then clustered (see `clustering`) in order to classify the relevant signal for the experiment.
    If `find_min` is set to `True` it finds minimum peaks of the input signal;
    `min_points` is the minimum number of points for a cluster to be considered relevant signal.
    Position of the relevant signal is returned.
    """

    x_ = np.unique(x)
    y_ = np.unique(y)
    # background removed over y axis
    z_ = z.reshape(len(y_), len(x_))

    z_ = -z_ if find_min else z_

    # masking
    z_masked = filter_data(z_)

    # renormalizing
    # z_masked_norm = scaling_signal(z_masked)
    z_masked_norm = scaling_slice(z_masked, axis=1)

    # filter data using find_peaks
    peaks_dict = peaks_finder(x_, y_, z_masked_norm)

    # normalizing peaks for clustering
    peaks = build_clustering_data(peaks_dict, z_masked)

    # clustering
    # In this function Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm is used;
    # HDBSCAN good for successfully capture clusters with different densities.
    hdb = HDBSCAN(copy=True, min_cluster_size=2)
    hdb.fit(peaks)
    labels = hdb.labels_

    # merging close clusters
    signal_classification = merging(
        peaks,
        labels,
        min_points,
        distance_xy=DISTANCE_XY,
        distance_z=DISTANCE_Z,
    )

    return peaks_dict["x"]["val"][signal_classification], peaks_dict["y"]["val"][
        signal_classification
    ]


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
