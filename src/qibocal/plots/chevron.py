import os.path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.plots.utils import get_data_subfolders


def landscape_2q_gate(folder, routine, qubit, format):
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    subfolder = get_data_subfolders(folder)[0]
    data = DataUnits.load_data(
        folder, subfolder, routine, format, f"data_q{lowfreq}{highfreq}"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V) - High Frequency",
            "MSR (V) - Low Frequency",  # TODO: change this to <Z>
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("theta", "rad")[data.df["q_freq"] == "high"][
                data.df["setup"] == "I"
            ].to_numpy(),
            y=data.get_values("MSR", "V")[data.df["q_freq"] == "high"][
                data.df["setup"] == "I"
            ].to_numpy(),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("theta", "rad")[data.df["q_freq"] == "high"][
                data.df["setup"] == "X"
            ].to_numpy(),
            y=data.get_values("MSR", "V")[data.df["q_freq"] == "high"][
                data.df["setup"] == "X"
            ].to_numpy(),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("theta", "rad")[data.df["q_freq"] == "low"][
                data.df["setup"] == "I"
            ].to_numpy(),
            y=data.get_values("MSR", "V")[data.df["q_freq"] == "low"][
                data.df["setup"] == "I"
            ].to_numpy(),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("theta", "rad")[data.df["q_freq"] == "low"][
                data.df["setup"] == "X"
            ].to_numpy(),
            y=data.get_values("MSR", "V")[data.df["q_freq"] == "low"][
                data.df["setup"] == "X"
            ].to_numpy(),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="theta (rad)",
        yaxis_title="MSR (V)",
        xaxis2_title="theta (rad)",
        yaxis2_title="MSR (V)",
    )
    return [fig]


def duration_amplitude_msr_flux_pulse(folder, routine, qubit, format):
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    subfolder = get_data_subfolders(folder)[0]
    data = DataUnits.load_data(
        folder, subfolder, routine, format, f"data_q{lowfreq}{highfreq}"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V) - High Frequency",
            "MSR (V) - Low Frequency",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            z=data.get_values("MSR", "V")[data.df["q_freq"] == "high"].to_numpy(),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            z=data.get_values("MSR", "V")[data.df["q_freq"] == "low"].to_numpy(),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="duration (ns)",
        yaxis_title="amplitude (dimensionless)",
        xaxis2_title="duration (ns)",
        yaxis2_title="amplitude (dimensionless)",
    )
    return [fig]


def duration_amplitude_I_flux_pulse(folder, routine, qubit, format):
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    subfolder = get_data_subfolders(folder)[0]
    data = DataUnits.load_data(
        folder, subfolder, routine, format, f"data_q{lowfreq}{highfreq}"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "I (V) - High Frequency",
            "I (V) - Low Frequency",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            z=data.get_values("i", "V")[data.df["q_freq"] == "high"].to_numpy(),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            z=data.get_values("i", "V")[data.df["q_freq"] == "low"].to_numpy(),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="duration (ns)",
        yaxis_title="amplitude (dimensionless)",
        xaxis2_title="duration (ns)",
        yaxis2_title="amplitude (dimensionless)",
    )
    return [fig]


def duration_amplitude_Q_flux_pulse(folder, routine, qubit, format):
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    subfolder = get_data_subfolders(folder)[0]
    data = DataUnits.load_data(
        folder, subfolder, routine, format, f"data_q{lowfreq}{highfreq}"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "Q (V) - High Frequency",
            "Q (V) - Low Frequency",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            z=data.get_values("q", "V")[data.df["q_freq"] == "high"].to_numpy(),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            z=data.get_values("q", "V")[data.df["q_freq"] == "low"].to_numpy(),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="duration (ns)",
        yaxis_title="amplitude (dimensionless)",
        xaxis2_title="duration (ns)",
        yaxis2_title="amplitude (dimensionless)",
    )
    return [fig]
