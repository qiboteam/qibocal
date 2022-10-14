# -*- coding: utf-8 -*-
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, Dataset
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


def frequency_msr_phase__fast_precision(folder, routine, qubit, format):
    try:
        data_fast = Dataset.load_data(folder, routine, format, f"fast_sweep_q{qubit}")
    except:
        data_fast = Dataset(quantities={"frequency": "Hz"})
    try:
        data_precision = Dataset.load_data(
            folder, routine, format, f"precision_sweep_q{qubit}"
        )
    except:
        data_precision = Dataset(quantities={"frequency": "Hz"})
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "label1",
                "label2",
            ]
        )

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
            x=data_fast.get_values("frequency", "GHz"),
            y=data_fast.get_values("MSR", "uV"),
            name="Fast",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_fast.get_values("frequency", "GHz"),
            y=data_fast.get_values("phase", "rad"),
            name="Fast",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=data_precision.get_values("frequency", "GHz"),
            y=data_precision.get_values("MSR", "uV"),
            name="Precision",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_precision.get_values("frequency", "GHz"),
            y=data_precision.get_values("phase", "rad"),
            name="Precision",
        ),
        row=1,
        col=2,
    )
    if len(data_fast) > 0 and len(data_fit) > 0:
        freqrange = np.linspace(
            min(data_fast.get_values("frequency", "GHz")),
            max(data_fast.get_values("frequency", "GHz")),
            2 * len(data_fast),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorenzian(
                    freqrange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"The estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} Hz.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"The estimated {params[1]} is {data_fit.df[params[1]][0]:.3f} uV.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
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


# For Rabi oscillations
def time_msr_phase(folder, routine, qubit, format):
    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(quantities={"Time": "ns"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "popt4",
                "label1",
                "label2",
            ]
        )

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

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        timerange = np.linspace(
            min(data.get_values("Time", "ns")),
            max(data.get_values("Time", "ns")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=timerange,
                y=rabi(
                    timerange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                    data_fit.get_values("popt4"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f} ns.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} uV.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    # last part
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

    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(quantities={"gain", "dimensionless"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "popt4",
                "label1",
                "label2",
            ]
        )

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

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        gainrange = np.linspace(
            min(data.get_values("gain", "dimensionless")),
            max(data.get_values("gain", "dimensionless")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=gainrange,
                y=rabi(
                    gainrange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                    data_fit.get_values("popt4"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.4f} uV",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gain (dimensionless)",
        yaxis_title="MSR (uV)",
    )
    return fig


def amplitude_msr_phase(folder, routine, qubit, format):

    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(quantities={"amplitude", "dimensionless"})
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Dataset()

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

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        amplituderange = np.linspace(
            min(data.get_values("amplitude", "dimensionless")),
            max(data.get_values("amplitude", "dimensionless")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=amplituderange,
                y=rabi(
                    amplituderange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                    data_fit.get_values("popt4"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.3f} uV.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.4f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="MSR (uV)",
    )
    return fig


# For Ramsey oscillations
def time_msr(folder, routine, qubit, format):
    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(name=f"data_q{qubit}", quantities={"wait": "ns", "t_max": "ns"})
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Dataset()

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("wait", "ns"),
            y=data.get_values("MSR", "uV"),
            name="Ramsey",
        ),
        row=1,
        col=1,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        timerange = np.linspace(
            min(data.get_values("wait", "ns")),
            max(data.get_values("wait", "ns")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=timerange,
                y=ramsey(
                    timerange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                    data_fit.get_values("popt4"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f} Hz.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} ns",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[2]} is {data_fit.df[params[2]][0]:.3f} Hz",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )
    return fig


# T1
def t1_time_msr_phase(folder, routine, qubit, format):
    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(quantities={"Time": "ns"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Dataset()

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
            name="T1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("phase", "rad"),
            name="T1",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        timerange = np.linspace(
            min(data.get_values("Time", "ns")),
            max(data.get_values("Time", "ns")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=timerange,
                y=exp(
                    timerange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} ns.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Time (ns)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# Flipping
def flips_msr_phase(folder, routine, qubit, format):
    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(quantities={"flips": "dimensionless"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Dataset()

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
            x=data.get_values("flips", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            name="Flipping MSR",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("flips", "dimensionless"),
            y=data.get_values("phase", "rad"),
            name="Flipping Phase",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        flipsrange = np.linspace(
            min(data.get_values("flips", "dimensionless")),
            max(data.get_values("flips", "dimensionless")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=flipsrange,
                y=flipping(
                    flipsrange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.4f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Flips (dimensionless)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Flips (dimensionless)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# For calibrate qubit states
def exc_gnd(folder, routine, qubit, format):

    try:
        parameters = Dataset.load_data(folder, routine, format, f"parameters_q{qubit}")
        rotation_angle = parameters.get_values("rotation_angle", "dimensionless")[0]
        threshold = parameters.get_values("threshold", "V")[0]
        fidelity = parameters.get_values("fidelity", "dimensionless")[0]
        assignment_fidelity = parameters.get_values(
            "assignment_fidelity", "dimensionless"
        )[0]
    except:
        parameters = Dataset(
            name=f"parameters_q{qubit}",
            quantities={
                "rotation_angle": "dimensionless",  # in degrees
                "threshold": "V",
                "fidelity": "dimensionless",
                "assignment_fidelity": "dimensionless",
            },
        )

    try:
        data_exc = Dataset.load_data(folder, routine, format, f"data_exc_q{qubit}")
    except:
        data_exc = Dataset(quantities={"iteration": "dimensionless"})

    fig = make_subplots(
        rows=1,
        cols=1,
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
            marker=dict(size=3, color="lightcoral"),
        ),
        row=1,
        col=1,
    )

    try:
        data_gnd = Dataset.load_data(folder, routine, format, f"data_gnd_q{qubit}")
    except:
        data_gnd = Dataset(quantities={"iteration": "dimensionless"})

    fig.add_trace(
        go.Scatter(
            x=data_gnd.get_values("i", "V"),
            y=data_gnd.get_values("q", "V"),
            name="gnd state",
            mode="markers",
            marker=dict(size=3, color="skyblue"),
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
            name=f" state1_voltage: {mod_iq_exc} <br> mean_state1: {iq_mean_exc}",
            mode="markers",
            marker=dict(size=10, color="red"),
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
            name=f" state0_voltage: {mod_iq_gnd} <br> mean_state0: {iq_mean_gnd}",
            mode="markers",
            marker=dict(size=10, color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="i (V)",
        yaxis_title="q (V)",
        width=1000,
    )

    title_text = f"""
    rotation_angle = {rotation_angle}<br>
    threshold = {threshold}<br>
    fidelity = {fidelity}<br>
    assignment_fidelity = {assignment_fidelity}
    """

    fig.update_xaxes(title_text=title_text, row=1, col=1)
    return fig


# allXY
def prob_gate(folder, routine, qubit, format):

    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(
            quantities={"probability": "dimensionless", "gateNumber": "dimensionless"}
        )

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY",),
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
        xaxis_title="Gate sequence number",
        yaxis_title="Z projection probability of qubit state |o>",
    )
    return fig


# allXY
def prob_gate_iteration(folder, routine, qubit, format):

    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(
            quantities={
                "probability": "dimensionless",
                "gateNumber": "dimensionless",
                "beta_param": "dimensionless",
            }
        )

    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY",),
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
        c = "#" + "{:06x}".format(n * 99999)
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
        xaxis_title="Gate sequence number",
        yaxis_title="Z projection probability of qubit state |o>",
    )
    return fig


# beta param tuning
def msr_beta(folder, routine, qubit, format):

    try:
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = Dataset(
            name=f"data_q{qubit}", quantities={"beta_param": "dimensionless"}
        )
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Dataset()

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
        subplot_titles=(f"beta_param_tuning",),
    )

    c = "#6597aa"
    fig.add_trace(
        go.Scatter(
            x=data.get_values("beta_param", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            line=dict(color=c),
            mode="markers",
            name="[Rx(pi/2) - Ry(pi)] - [Ry(pi/2) - Rx(pi)]",
        ),
        row=1,
        col=1,
    )
    # add fitting traces
    if len(data) > 0 and len(data_fit) > 0:
        beta_param = np.linspace(
            min(data.get_values("beta_param", "dimensionless")),
            max(data.get_values("beta_param", "dimensionless")),
            20,
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=beta_param,
                y=cos(
                    beta_param,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.4f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Beta parameter",
        yaxis_title="MSR[uV]",
    )
    return fig


def dispersive_frequency_msr_phase(folder, routine, qubit, formato):

    try:
        data_spec = Dataset.load_data(folder, routine, formato, f"data_q{qubit}")
    except:
        data_spec = Dataset(name=f"data_q{qubit}", quantities={"frequency": "Hz"})

    try:
        data_shifted = Dataset.load_data(
            folder, routine, formato, f"data_shifted_q{qubit}"
        )
    except:
        data_shifted = Dataset(
            name=f"data_shifted_q{qubit}", quantities={"frequency": "Hz"}
        )

    try:
        data_fit = Data.load_data(folder, routine, formato, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "label1",
                "label2",
            ]
        )

    try:
        data_fit_shifted = Data.load_data(
            folder, routine, formato, f"fit_shifted_q{qubit}"
        )
    except:
        data_fit_shifted = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "label1",
                "label2",
            ]
        )

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
            x=data_spec.get_values("frequency", "GHz"),
            y=data_spec.get_values("MSR", "uV"),
            name="Spectroscopy",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_spec.get_values("frequency", "GHz"),
            y=data_spec.get_values("phase", "rad"),
            name="Spectroscopy",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=data_shifted.get_values("frequency", "GHz"),
            y=data_shifted.get_values("MSR", "uV"),
            name="Shifted Spectroscopy",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data_shifted.get_values("frequency", "GHz"),
            y=data_shifted.get_values("phase", "rad"),
            name="Shifted Spectroscopy",
        ),
        row=1,
        col=2,
    )

    # fitting traces
    if len(data_spec) > 0 and len(data_fit) > 0:
        freqrange = np.linspace(
            min(data_spec.get_values("frequency", "GHz")),
            max(data_spec.get_values("frequency", "GHz")),
            2 * len(data_spec),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorenzian(
                    freqrange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                ),
                name="Fit spectroscopy",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"The estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} Hz.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    # fitting shifted  traces
    if len(data_shifted) > 0 and len(data_fit_shifted) > 0:
        freqrange = np.linspace(
            min(data_shifted.get_values("frequency", "GHz")),
            max(data_shifted.get_values("frequency", "GHz")),
            2 * len(data_shifted),
        )
        params = [i for i in list(data_fit_shifted.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorenzian(
                    freqrange,
                    data_fit_shifted.get_values("popt0"),
                    data_fit_shifted.get_values("popt1"),
                    data_fit_shifted.get_values("popt2"),
                    data_fit_shifted.get_values("popt3"),
                ),
                name="Fit shifted spectroscopy",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"The estimated shifted {params[0]} is {data_fit_shifted.df[params[0]][0]:.1f} Hz.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )
    return fig
