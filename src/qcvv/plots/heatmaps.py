# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qcvv.data import Dataset


def frequency_flux_msr_phase(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (deg)",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("current", "A"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("current", "A"),
            z=data.get_values("phase", "deg"),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Current (A)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Current (A)",
    )
    return fig


def frequency_attenuation_msr_phase(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (deg)",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("attenuation", "dB"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("attenuation", "dB"),
            z=data.get_values("phase", "deg"),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Attenuation (dB)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Attenuation (dB)",
    )
    return fig


def duration_gain_msr_phase(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (deg)",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("duration", "ns"),
            y=data.get_values("gain", "db"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("duration", "ns"),
            y=data.get_values("gain", "db"),
            z=data.get_values("phase", "deg"),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="duration (ns)",
        yaxis_title="gain (db)",
        xaxis2_title="duration (ns)",
        yaxis2_title="gain (db)",
    )
    return fig


def duration_amplitude_msr_phase(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (deg)",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("duration", "ns"),
            y=data.get_values("amplitude", "V"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("duration", "ns"),
            y=data.get_values("amplitude", "V"),
            z=data.get_values("phase", "deg"),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="duration (ns)",
        yaxis_title="amplitude (V)",
        xaxis2_title="duration (ns)",
        yaxis2_title="amplitude (V)",
    )
    return fig
