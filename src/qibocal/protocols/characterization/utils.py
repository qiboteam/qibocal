import statistics
from enum import Enum

import lmfit
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.auto.operation import Results
from qibocal.config import log
from qibocal.plots.utils import get_color


class PowerLevel(Enum):
    """Power Regime for Resonator Spectroscopy"""

    high = "high"
    low = "low"


def lorentzian(frequency, amplitude, center, sigma, offset):
    # http://openafox.com/science/peak-function-derivations.html
    return (amplitude / np.pi) * (
        sigma / ((frequency - center) ** 2 + sigma**2)
    ) + offset


def lorentzian_fit(data, qubit):
    qubit_data = (
        data.df[data.df["qubit"] == qubit].drop(columns=["qubit"]).reset_index()
    )
    frequencies = qubit_data["frequency"].pint.to("GHz").pint.magnitude
    voltages = qubit_data["MSR"].pint.to("uV").pint.magnitude
    model_Q = lmfit.Model(lorentzian)

    # Guess parameters for Lorentzian max or min
    if (
        data.resonator_type == "3D"
        and data.__class__.__name__ == "ResonatorSpectroscopyData"
    ) or (
        data.resonator_type == "2D"
        and data.__class__.__name__ == "QubitSpectroscopyData"
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
        subplot_titles=(
            "MSR (uV)",
            "phase (rad)",
        ),
    )
    qubit_data = data.df[data.df["qubit"] == qubit].drop(columns=["i", "q", "qubit"])

    fitting_report = ""

    frequencies = qubit_data["frequency"].pint.to("GHz").pint.magnitude.unique()
    fig.add_trace(
        go.Scatter(
            x=qubit_data["frequency"].pint.to("GHz").pint.magnitude,
            y=qubit_data["MSR"].pint.to("uV").pint.magnitude,
            marker_color=get_color(0),
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
            x=qubit_data["frequency"].pint.to("GHz").pint.magnitude,
            y=qubit_data["phase"].pint.to("rad").pint.magnitude,
            marker_color=get_color(1),
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
            marker_color=get_color(2),
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

    fitting_report += f"{qubit} | {label}: {freq[qubit]*1e9:,.0f} Hz<br>"

    if fit.amplitude:
        fitting_report += f"{qubit} | amplitude: {fit.amplitude[qubit]} <br>"

    if fit.attenuation:
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


def find_min_msr(data, resonator_type, fit_type):
    # Find the minimum values of z for each level of attenuation and their locations (x, y).
    data = data[["frequency", fit_type, "MSR"]].to_numpy()
    if resonator_type == "3D":
        func = np.argmax
    else:
        func = np.argmin
    min_msr_per_attenuation = []
    for i in np.unique(data[:, 1]):
        selected = data[data[:, 1] == i]
        min_msr_per_attenuation.append(selected[func(selected[:, 2])])

    return np.array(min_msr_per_attenuation)


def get_max_freq(distribution_points):
    freqs = [point[0] for point in distribution_points]
    max_freq = statistics.mode(freqs)
    return max_freq


def get_points_with_max_freq(min_points, max_freq):
    matching_points = [point for point in min_points if point[0] == max_freq]
    if matching_points:
        return max(matching_points, key=lambda point: point[1]), min(
            matching_points, key=lambda point: point[1]
        )
    x_values = [point[0] for point in min_points]
    closest_idx = np.argmin(np.abs(np.array(x_values) - max_freq))
    closest_point = min_points[closest_idx]
    matching_points = [point for point in min_points if point[0] == closest_point[0]]
    return max(matching_points, key=lambda point: point[1]), min(
        matching_points, key=lambda point: point[1]
    )


def norm(x_mags):
    return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))
