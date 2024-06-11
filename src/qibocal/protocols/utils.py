from colorsys import hls_to_rgb
from enum import Enum
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.qubits import QubitId
from scipy import constants
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Results
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


def effective_qubit_temperature(
    prob_0: np.array, prob_1: np.array, qubit_frequency: float, nshots: int
):
    """Calculates the qubit effective temperature.

    The formula used is the following one:

    kB Teff = - hbar qubit_freq / ln(prob_1/prob_0)

    Args:
        prob_0 (np.array): population 0 samples
        prob_1 (np.array): population 1 samples
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


def calculate_frequencies(results, qubit_list):
    """Calculates outcome frequencies from individual shots.
    Args:
        results (dict): return of execute_pulse_sequence
        qubit_list (list): list of qubit ids executed in pulse sequence.

    Returns:
        dictionary containing frequencies.
    """
    shots = np.stack([results[i].samples for i in qubit_list]).T
    values, counts = np.unique(shots, axis=0, return_counts=True)

    return {"".join(str(int(i)) for i in v): cnt for v, cnt in zip(values, counts)}


class PowerLevel(str, Enum):
    """Power Regime for Resonator Spectroscopy"""

    high = "high"
    low = "low"


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


def spectroscopy_plot(data, qubit, fit: Results = None):
    figures = []
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    qubit_data = data[qubit]
    fitting_report = ""
    frequencies = qubit_data.freq * HZ_TO_GHZ
    signal = qubit_data.signal

    phase = qubit_data.phase
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=signal,
            opacity=1,
            name="Frequency",
            showlegend=True,
            legendgroup="Frequency",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=phase,
            opacity=1,
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
        ),
        row=1,
        col=2,
    )

    show_error_bars = not np.isnan(qubit_data.error_signal).any()
    if show_error_bars:
        errors_signal = qubit_data.error_signal
        errors_phase = qubit_data.error_phase
        fig.add_trace(
            go.Scatter(
                x=np.concatenate((frequencies, frequencies[::-1])),
                y=np.concatenate(
                    (signal + errors_signal, (signal - errors_signal)[::-1])
                ),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Signal Errors",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate((frequencies, frequencies[::-1])),
                y=np.concatenate((phase + errors_phase, (phase - errors_phase)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Phase Errors",
            ),
            row=1,
            col=2,
        )

    freqrange = np.linspace(
        min(frequencies),
        max(frequencies),
        2 * len(frequencies),
    )

    if fit is not None:
        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorentzian(freqrange, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        if data.power_level is PowerLevel.low:
            label = "readout frequency[Hz]"
            freq = fit.frequency
        elif data.power_level is PowerLevel.high:
            label = "bare resonator frequency[Hz]"
            freq = fit.bare_frequency
        else:
            label = "qubit frequency[Hz]"
            freq = fit.frequency

        if data.attenuations:
            if data.attenuations[qubit] is not None:
                if show_error_bars:
                    labels = [label, "amplitude", "attenuation", "chi2 reduced"]
                    values = [
                        (
                            freq[qubit],
                            fit.error_fit_pars[qubit][1],
                        ),
                        (data.amplitudes[qubit], 0),
                        (data.attenuations[qubit], 0),
                        fit.chi2_reduced[qubit],
                    ]
                else:
                    labels = [label, "amplitude", "attenuation"]
                    values = [
                        freq[qubit],
                        data.amplitudes[qubit],
                        data.attenuations[qubit],
                    ]
        if data.amplitudes[qubit] is not None:
            if show_error_bars:
                labels = [label, "amplitude", "chi2 reduced"]
                values = [
                    (
                        freq[qubit],
                        fit.error_fit_pars[qubit][1],
                    ),
                    (data.amplitudes[qubit], 0),
                    fit.chi2_reduced[qubit],
                ]
            else:
                labels = [label, "amplitude"]
                values = [freq[qubit], data.amplitudes[qubit]]

            fitting_report = table_html(
                table_dict(
                    qubit,
                    labels,
                    values,
                    display_error=show_error_bars,
                )
            )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHz]",
        yaxis_title="Signal [a.u.]",
        xaxis2_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
    )
    figures.append(fig)

    return figures, fitting_report


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
                f"{np.format_float_positional(round(error*10**(-1*magnitude), ndigits), trim = '-')}e{magnitude}"
            )
        else:
            rounded_values.append(f"{round(value * 10 ** (-1 * magnitude), ndigits)}")
            rounded_errors.append(
                f"{np.format_float_positional(round(error*10**(-1*magnitude), ndigits), trim = '-')}"
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
    observed: npt.NDArray,
    estimated: npt.NDArray,
    errors: npt.NDArray,
    dof: float = None,
):
    if np.count_nonzero(errors) < len(errors):
        return EXTREME_CHI

    if dof is None:
        dof = len(observed) - 1

    chi2 = np.sum(np.square((observed - estimated) / errors)) / dof

    return chi2


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
    data: npt.NDArray,
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
            title_text=f"i [a.u.]",
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
            # pylint: disable=E1101
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
    second_mask = z[first_mask] < max if feat == "min" else z[first_mask] > min
    return x[first_mask][second_mask], y[first_mask][second_mask]
