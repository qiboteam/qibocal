import statistics
from enum import Enum, auto

import lmfit
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...auto.operation import Results
from ...config import log
from ...data import DataUnits
from ...plots.utils import get_color


class PowerLevel(Enum):
    high = auto()
    low = auto()


def lorentzian(frequency, amplitude, center, sigma, offset):
    # http://openafox.com/science/peak-function-derivations.html
    return (amplitude / np.pi) * (
        sigma / ((frequency - center) ** 2 + sigma**2)
    ) + offset


def lorentzian_fit(data: DataUnits) -> list:
    qubits = data.df["qubit"].unique()
    resonator_type = data.df["resonator_type"].unique()

    power_level = data.df["power_level"].unique() if "power_level" in data.df else None

    bare_frequency = {}
    amplitudes = {}
    frequency = {}
    fitted_parameters = {}

    for qubit in qubits:
        drop_columns = [
            "qubit",
            "iteration",
            "resonator_type",
            "amplitude",
        ]
        if power_level is not None:
            drop_columns += ["power_level"]
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=drop_columns)
            .groupby("frequency", as_index=False)
            .mean()
        )

        frequencies = qubit_data["frequency"].pint.to("GHz").pint.magnitude

        voltages = qubit_data["MSR"].pint.to("uV").pint.magnitude

        model_Q = lmfit.Model(lorentzian)

        if resonator_type == "3D":
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
            f0 = fit_res.best_values["center"]
            BW = fit_res.best_values["sigma"] * 2
            Q = abs(f0 / BW)
            peak_voltage = (
                fit_res.best_values["amplitude"]
                / (fit_res.best_values["sigma"] * np.pi)
                + fit_res.best_values["offset"]
            )
            freq = f0

        except:
            log.warning("lorentzian_fit: the fitting was not successful")

        frequency[qubit] = f0

        if power_level == "high":  # TODO: fix this in PowerLevel.low
            bare_frequency[qubit] = f0
        data_df = data.df
        amplitude = data_df[data_df.qubit == qubit]["amplitude"].unique()
        amplitudes[qubit] = amplitude[0]
        fitted_parameters[qubit] = fit_res.best_values

    if power_level is not None:
        output = {
            "frequency": frequency,
            "fitted_parameters": fitted_parameters,
            "bare_frequency": bare_frequency,
            "amplitude": amplitudes,
        }
    else:
        output = {
            "frequency": frequency,
            "fitted_parameters": fitted_parameters,
            "amplitude": amplitudes,
        }
    return output


def spectroscopy_plot(data: DataUnits, fit: Results, qubit):
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

    power_level = data.df["power_level"].unique() if "power_level" in data.df else None
    drop_columns = ["i", "q", "qubit"]
    if power_level is not None:
        drop_columns += ["power_level"]

    data.df = data.df[data.df["qubit"] == qubit].drop(columns=drop_columns)
    iterations = data.df["iteration"].unique()

    fitting_report = ""
    report_n = 0

    if len(iterations) > 1:
        opacity = 0.3
    else:
        opacity = 1
    for iteration in iterations:
        frequencies = data.df["frequency"].pint.to("GHz").pint.magnitude.unique()
        iteration_data = data.df[data.df["iteration"] == iteration]
        fig.add_trace(
            go.Scatter(
                x=iteration_data["frequency"].pint.to("GHz").pint.magnitude,
                y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                marker_color=get_color(2 * report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}: Data",
                showlegend=not bool(iteration),
                legendgroup=f"q{qubit}/r{report_n}: Data",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=iteration_data["frequency"].pint.to("GHz").pint.magnitude,
                y=iteration_data["phase"].pint.to("rad").pint.magnitude,
                marker_color=get_color(2 * report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}: Data",
                showlegend=False,
                legendgroup=f"q{qubit}/r{report_n}: Data",
            ),
            row=1,
            col=2,
        )
    if len(iterations) > 1:
        data.df = data.df.drop(columns=["iteration"])  # pylint: disable=E1101
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=data.df.groupby("frequency")["MSR"]
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                marker_color=get_color(2 * report_n),
                name=f"q{qubit}/r{report_n}: Average",
                showlegend=True,
                legendgroup=f"q{qubit}/r{report_n}: Average",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=data.df.groupby("frequency")["phase"]
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                marker_color=get_color(2 * report_n),
                showlegend=False,
                legendgroup=f"q{qubit}/r{report_n}: Average",
            ),
            row=1,
            col=2,
        )
    if len(data) > 0:
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
                name=f"q{qubit}/r{report_n} Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(4 * report_n + 2),
            ),
            row=1,
            col=1,
        )

        if power_level == "low":  # TODO:change this to PowerLevel.low
            label = "readout frequency"
            freq = fit.frequency
        elif power_level == "high":
            label = "bare resonator frequency"
            freq = fit.bare_frequency
        else:
            label = "qubit frequency"
            freq = fit.frequency
        fitting_report += (
            f"q{qubit}/r{report_n} | {label}: {freq[qubit]*1e9:,.0f} Hz<br>"
        )

        if fit.amplitude:
            fitting_report += (
                f"q{qubit}/r{report_n} | amplitude: {fit.amplitude[qubit]} <br>"
            )

        if fit.attenuation:
            fitting_report += (
                f"q{qubit}/r{report_n} | attenuation: {fit.attenuation[qubit]} <br>"
            )
        fig.add_vline(x=np.mean(frequencies))

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
