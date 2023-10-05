from colorsys import hls_to_rgb
from enum import Enum
from typing import Optional

import lmfit
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from numba import njit
from plotly.subplots import make_subplots
from qibolab.qubits import QubitId
from scipy.stats import mode

from qibocal.auto.operation import Data, Results
from qibocal.config import log

GHZ_TO_HZ = 1e9
HZ_TO_GHZ = 1e-9
V_TO_UV = 1e6
S_TO_NS = 1e9
EXTREME_CHI = 1e4
"""Chi2 output when errors list contains zero elements"""


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
    voltages = data.msr * V_TO_UV
    model_Q = lmfit.Model(lorentzian)

    # Guess parameters for Lorentzian max or min
    # TODO: probably this is not working on HW
    if (resonator_type == "3D" and fit == "resonator") or (
        resonator_type == "2D" and fit == "qubit"
    ):
        guess_center = frequencies[
            np.argmax(voltages)
        ]  # Argmax = Returns the indices of the maximum values along an axis.
        guess_offset = np.mean(
            voltages[np.abs(voltages - np.mean(voltages) < np.std(voltages))]
        )
        guess_sigma = abs(frequencies[np.argmin(voltages)] - guess_center)
        guess_amp = (np.max(voltages) - guess_offset) * guess_sigma * np.pi

    else:
        guess_center = frequencies[
            np.argmin(voltages)
        ]  # Argmin = Returns the indices of the minimum values along an axis.
        guess_offset = np.mean(
            voltages[np.abs(voltages - np.mean(voltages) < np.std(voltages))]
        )
        guess_sigma = abs(frequencies[np.argmax(voltages)] - guess_center)
        guess_amp = (np.min(voltages) - guess_offset) * guess_sigma * np.pi

    # Add guessed parameters to the model
    model_Q.set_param_hint("center", value=guess_center, vary=True)
    model_Q.set_param_hint("sigma", value=guess_sigma, vary=True)
    model_Q.set_param_hint("amplitude", value=guess_amp, vary=True)
    model_Q.set_param_hint("offset", value=guess_offset, vary=True)
    guess_parameters = model_Q.make_params()

    # fit the model with the data and guessed parameters
    try:
        fit_res = model_Q.fit(
            data=voltages, frequency=frequencies, params=guess_parameters
        )
        # get the values for postprocessing and for legend.
        return fit_res.best_values["center"], fit_res.best_values

    except:
        log.warning("lorentzian_fit: the fitting was not successful")
        fit_res = lmfit.model.ModelResult(model=model_Q, params=guess_parameters)
        return guess_center, fit_res.params.valuesdict()


def spectroscopy_plot(data, qubit, fit: Results = None):
    figures = []
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    qubit_data = data[qubit]
    fitting_report = None

    frequencies = qubit_data.freq * HZ_TO_GHZ
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=qubit_data.msr * 1e6,
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
            y=qubit_data.phase,
            opacity=1,
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
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
                y=lorentzian(freqrange, **params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        if data.power_level is PowerLevel.low:
            label = "readout frequency"
            freq = fit.frequency
        elif data.power_level is PowerLevel.high:
            label = "bare resonator frequency"
            freq = fit.bare_frequency
        else:
            label = "qubit frequency"
            freq = fit.frequency

        fitting_report = f"{qubit} | {label}: {freq[qubit]*GHZ_TO_HZ:,.0f} Hz<br>"

        if fit.amplitude[qubit] is not None:
            fitting_report += f"{qubit} | amplitude: {fit.amplitude[qubit]} <br>"

        if data.__class__.__name__ == "ResonatorSpectroscopyAttenuationData":
            if fit.attenuation[qubit] is not None and fit.attenuation[qubit] != 0:
                fitting_report += (
                    f"{qubit} | attenuation: {fit.attenuation[qubit]} <br>"
                )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )
    figures.append(fig)

    return figures, fitting_report


def norm(x_mags):
    return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))


@njit(["float64[:] (float64[:], float64[:])"], parallel=True, cache=True)
def cumulative(input_data, points):
    r"""Evaluates in data the cumulative distribution
    function of `points`.
    WARNING: `input_data` and `points` should be sorted data.
    """
    input_data = np.sort(input_data)
    points = np.sort(points)
    # data and points sorted
    prob = []
    app = 0

    for val in input_data:
        app += np.maximum(np.searchsorted(points[app::], val), 0)
        prob.append(float(app))

    return np.array(prob)


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
        freqs = np.unique(qubit_data.freq)
        nvalues = len(np.unique(qubit_data[fit_type]))
        nfreq = len(freqs)
        msrs = np.reshape(qubit_data.msr, (nvalues, nfreq))
        if data.resonator_type == "3D":
            peak_freqs = freqs[np.argmax(msrs, axis=1)]
        else:
            peak_freqs = freqs[np.argmin(msrs, axis=1)]

        max_freq = np.max(peak_freqs)
        min_freq = np.min(peak_freqs)
        middle_freq = (max_freq + min_freq) / 2

        freq_hp = peak_freqs[peak_freqs < middle_freq]
        freq_lp = peak_freqs[peak_freqs >= middle_freq]

        freq_hp = mode(freq_hp, keepdims=True)[0]
        freq_lp = mode(freq_lp, keepdims=True)[0]

        if fit_type == "amp":
            if data.resonator_type == "3D":
                ro_val = getattr(qubit_data, fit_type)[
                    np.argmax(qubit_data.msr[np.where(qubit_data.freq == freq_lp)[0]])
                ]
            else:
                ro_val = getattr(qubit_data, fit_type)[
                    np.argmin(qubit_data.msr[np.where(qubit_data.freq == freq_lp)[0]])
                ]
        else:
            high_att_max = np.max(
                getattr(qubit_data, fit_type)[np.where(qubit_data.freq == freq_hp)[0]]
            )
            high_att_min = np.min(
                getattr(qubit_data, fit_type)[np.where(qubit_data.freq == freq_hp)[0]]
            )

            ro_val = round((high_att_max + high_att_min) / 2)
            ro_val = ro_val + (ro_val % 2)

        low_freqs[qubit] = freq_lp.item() * HZ_TO_GHZ
        high_freqs[qubit] = freq_hp[0] * HZ_TO_GHZ
        ro_values[qubit] = ro_val
    return [low_freqs, high_freqs, ro_values]


def eval_magnitude(value):
    """number of non decimal digits in `value`"""
    if value == 0 or not np.isfinite(value):
        return 0
    return int(np.floor(np.log10(abs(value))))


def fill_table(
    qubit: QubitId,
    name: str,
    value: float,
    error: Optional[float],
    unit: str = None,
    ndigits: int = 2,
) -> str:
    """
    Return a row of the report table with the correct number of
    significant digits.

    Args:
        qubit (QubitId): Qubit.
        name (str): Variable's name.
        value (float): Variable's value.
        error (float): Error associated to the variable.
        unit (str): Measurement unit. Default value `None`.
        ndigits (int): Number of decimal digits to display when error is `None`
            (i.e. it is not evaluated).
    """
    row = f"{qubit}| {name}: "
    if value:
        magnitude = eval_magnitude(value)
    else:
        magnitude = 0
    if error:
        ndigits = max(significant_digit(error * 10 ** (-1 * magnitude)), 0)
        row += f"({round(value*10**(-1*magnitude), ndigits)} ± {np.format_float_positional(round(error*10**(-1*magnitude), ndigits), trim = '-')})"
    else:
        row += f"{round(value*10**(-1* magnitude), ndigits)}"
    if magnitude != 0:
        row += f"* 10^{magnitude}"
    row += f" {unit} <br>" if unit else f"<br>"
    return row


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

    if np.isinf(np.real(number)) or np.real(number) >= 1 or number == 0:
        return -1

    position = max(np.ceil(-np.log10(abs(np.real(number)))), -1)

    if np.imag(number) != 0:
        position = max(position, np.ceil(-np.log10(abs(np.imag(number)))))

    return int(position)
