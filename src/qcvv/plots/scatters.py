# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qcvv.data import Dataset


def frequency_msr_phase__fast_precision(folder, routine, qubit, formato):

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    import os.path

    file_fast = f"{folder}/data/{routine}/fast_sweep_q{qubit}.csv"
    if os.path.exists(file_fast):
        data_fast = Dataset.load_data(folder, routine, formato, f"fast_sweep_q{qubit}")

        fig.add_trace(
            go.Scatter(
                x=data_fast.get_values("frequency", "Hz"),
                y=data_fast.get_values("MSR", "uV"),
                name="Fast",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data_fast.get_values("frequency", "Hz"),
                y=data_fast.get_values("phase", "rad"),
                name="Fast",
            ),
            row=1,
            col=2,
        )

    file_precision = f"{folder}/data/{routine}/precision_sweep_q{qubit}.csv"
    if os.path.exists(file_precision):
        data_precision = Dataset.load_data(
            folder, routine, formato, f"precision_sweep_q{qubit}"
        )

        fig.add_trace(
            go.Scatter(
                x=data_precision.get_values("frequency", "Hz"),
                y=data_precision.get_values("MSR", "uV"),
                name="Precision",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data_precision.get_values("frequency", "Hz"),
                y=data_precision.get_values("phase", "rad"),
                name="Precision",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (Hz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Phase (rad)",
    )
    return fig


def frequency_attenuation_msr_phase__cut(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    plot1d_attenuation = 30  # attenuation value to use for 1D frequency vs MSR plot

    fig = go.Figure()
    # index data on a specific attenuation value
    smalldf = data.df[data.get_values("attenuation", "dB") == plot1d_attenuation].copy()
    # split multiple software averages to different datasets
    datasets = []
    while len(smalldf):
        datasets.append(smalldf.drop_duplicates("frequency"))
        smalldf.drop(datasets[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets[-1]["frequency"].pint.to("GHz").pint.magnitude,
                y=datasets[-1]["MSR"].pint.to("V").pint.magnitude,
            ),
        )

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting,
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (V)",
    )
    return fig


def frequency_msr_phase(folder, routine, qubit, format):
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )
    import os.path

    file = f"{folder}/data/{routine}/data_q{qubit}.csv"
    if os.path.exists(file):
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
        data.df.sort_values(by="frequency", inplace=True)
        fig.add_trace(
            go.Scatter(
                x=data.get_values("frequency", "Hz"),
                y=data.get_values("MSR", "uV"),
                name="MSR",
                mode="lines+markers",
                marker=dict(size=5, color="LightSeaGreen"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.get_values("frequency", "Hz"),
                y=data.get_values("phase", "rad"),
                name="phase",
                mode="lines+markers",
                marker=dict(size=5, color="LightSeaGreen"),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting,
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (V)",
    )
    return fig


# For Rabi oscillations
def time_msr_phase(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("MSR", "uV"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("phase", "rad"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Time (ns)",
        yaxis2_title="Phase (rad)",
    )
    return fig


def gain_msr_phase(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("gain", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("gain", "dimensionless"),
            y=data.get_values("phase", "rad"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gain (unitless)",
        yaxis_title="MSR (uV)",
    )
    return fig


def amplitude_msr_phase(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("amplitude", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("amplitude", "dimensionless"),
            y=data.get_values("phase", "rad"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (unitless)",
        yaxis_title="MSR (uV)",
    )
    return fig


def ro_phase_msr(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("RO_pulse_phase", "deg"),
            y=data.get_values("MSR", "uV"),
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="RO_pulse_phase (deg)",
        yaxis_title="MSR (uV)",
    )
    return fig


# For Ramsey oscillations
def time_msr(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("wait", "ns"),
            y=data.get_values("MSR", "uV"),
            name="Ramsey detuned",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )
    return fig


# For calibrate qubit states
def exc_gnd(folder, routine, qubit, formato):

    import os.path

    file_exc = f"{folder}/data/{routine}/data_exc_q{qubit}.csv"
    if os.path.exists(file_exc):
        data_exc = Dataset.load_data(folder, routine, formato, f"data_exc_q{qubit}")

        fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=("Calibrate qubit states",),
        )

        fig.add_trace(
            go.Scatter(
                x=data_exc.get_values("i", "V"),
                y=data_exc.get_values("q", "V"),
                name="exc_state",
                mode="markers",
            ),
            row=1,
            col=1,
        )

        i_exc = data_exc.get_values("i", "V")
        q_exc = data_exc.get_values("q", "V")

        i_mean_exc = i_exc.mean()
        q_mean_exc = q_exc.mean()
        iq_mean_exc = complex(i_mean_exc, q_mean_exc)
        mod_iq_exc = abs(iq_mean_exc) * 1e6

        fig.add_trace(
            go.Scatter(
                x=[i_mean_exc],
                y=[q_mean_exc],
                name=f" state1_voltage: {mod_iq_exc} <br> mean_exc_state: {iq_mean_exc}",
                mode="markers",
                marker_size=16,
            ),
            row=1,
            col=1,
        )

    file_gnd = f"{folder}/data/{routine}/data_gnd_q{qubit}.csv"
    if os.path.exists(file_gnd):
        data_gnd = Dataset.load_data(folder, routine, formato, f"data_gnd_q{qubit}")

        fig.add_trace(
            go.Scatter(
                x=data_gnd.get_values("i", "V"),
                y=data_gnd.get_values("q", "V"),
                name="gnd state",
                mode="markers",
            ),
            row=1,
            col=1,
        )

        i_gnd = data_gnd.get_values("i", "V")
        q_gnd = data_gnd.get_values("q", "V")

        i_mean_gnd = i_gnd.mean()
        q_mean_gnd = q_gnd.mean()
        iq_mean_gnd = complex(i_mean_gnd, q_mean_gnd)
        mod_iq_gnd = abs(iq_mean_gnd) * 1e6

        fig.add_trace(
            go.Scatter(
                x=[i_mean_gnd],
                y=[q_mean_gnd],
                name=f" state0_voltage: {mod_iq_gnd} <br> mean_gnd_state: {iq_mean_gnd}",
                mode="markers",
                marker_size=16,
            ),
            row=1,
            col=1,
        )

        fig.update_layout(
            showlegend=True,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="i (V)",
            yaxis_title="q (V)",
        )

    return fig


# allXY
def prob_gate(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY_qubit{qubit}",),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("gateNumber", "dimensionless"),
            y=data.get_values("probability", "dimensionless"),
            mode="markers",
            name="Probabilities",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gate sequene number",
        yaxis_title="Z projection probability of qubit state |o>",
    )
    return fig


# allXY
def prob_gate_iteration(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY_qubit{qubit}",),
    )

    gates = len(data.get_values("gateNumber", "dimensionless"))
    # print(gates)
    import numpy as np

    for n in range(gates // 21):
        data_start = n * 21
        data_end = data_start + 21
        beta_param = np.array(data.get_values("beta_param", "dimensionless"))[
            data_start
        ]
        gates = np.array(data.get_values("gateNumber", "dimensionless"))[
            data_start:data_end
        ]
        probabilities = np.array(data.get_values("probability", "dimensionless"))[
            data_start:data_end
        ]
        c = "#" + "{:06x}".format(n * 823000)
        fig.add_trace(
            go.Scatter(
                x=gates,
                y=probabilities,
                mode="markers+lines",
                line=dict(color=c),
                name=f"beta_parameter = {beta_param}",
                marker_size=16,
            ),
            row=1,
            col=1,
        )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gate sequene number",
        yaxis_title="Z projection probability of qubit state |o>",
    )
    return fig
