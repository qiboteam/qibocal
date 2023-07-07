from enum import Enum

import lmfit
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.auto.operation import Results
from qibocal.config import log

GHZ_TO_HZ = 1e9
HZ_TO_GHZ = 1e-9
V_TO_UV = 1e6
S_TO_NS = 1e9


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


def spectroscopy_plot(data, fit: Results, qubit):
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

    fitting_report += f"{qubit} | {label}: {freq[qubit]*GHZ_TO_HZ:,.0f} Hz<br>"

    if fit.amplitude[qubit] is not None:
        fitting_report += f"{qubit} | amplitude: {fit.amplitude[qubit]} <br>"
        if data.power_level is PowerLevel.high:
            # TODO: find better solution for not updating amplitude in high power
            fit.amplitude.pop(qubit)

    if data.__class__.__name__ == "ResonatorSpectroscopyAttenuationData":
        if fit.attenuation[qubit] is not None and fit.attenuation[qubit] != 0:
            fitting_report += f"{qubit} | attenuation: {fit.attenuation[qubit]} <br>"

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
def cum_method(input_data, points):
    # data and points sorted
    input_data = np.sort(input_data)
    points = np.sort(points)

    prob = []
    app = 0.0
    for val in input_data:
        app += np.maximum(np.searchsorted(points[app::], val), 0)
        prob.append(app)

    return np.array(prob)
