import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


def frequency_msr_phase__fast_precision(folder, routine, qubit, format):
    try:
        data_fast = DataUnits.load_data(folder, routine, format, f"fast_sweep_q{qubit}")
    except:
        data_fast = DataUnits(quantities={"frequency": "Hz"})
    try:
        data_precision = DataUnits.load_data(
            folder, routine, format, f"precision_sweep_q{qubit}"
        )
    except:
        data_precision = DataUnits(quantities={"frequency": "Hz"})
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
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
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
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"Time": "ns"})

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
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"gain", "dimensionless"})

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
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"amplitude", "dimensionless"})
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

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
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(
            name=f"data_q{qubit}", quantities={"wait": "ns", "t_max": "ns"}
        )
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

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
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"Time": "ns"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

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
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"flips": "dimensionless"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

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
        data_exc = DataUnits.load_data(folder, routine, format, f"data_exc_q{qubit}")
    except:
        data_exc = DataUnits(quantities={"iteration": "dimensionless"})

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
        data_gnd = DataUnits.load_data(folder, routine, format, f"data_gnd_q{qubit}")
    except:
        data_gnd = DataUnits(quantities={"iteration": "dimensionless"})

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

    return fig


# allXY
def prob_gate(folder, routine, qubit, format):

    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(
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
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(
            quantities={
                "probability": "dimensionless",
                "gateNumber": "dimensionless",
                "beta_param": "dimensionless",
            }
        )

    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
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
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(
            name=f"data_q{qubit}", quantities={"beta_param": "dimensionless"}
        )
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

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
        data_spec = DataUnits.load_data(folder, routine, formato, f"data_q{qubit}")
    except:
        data_spec = DataUnits(name=f"data_q{qubit}", quantities={"frequency": "Hz"})

    try:
        data_shifted = DataUnits.load_data(
            folder, routine, formato, f"data_shifted_q{qubit}"
        )
    except:
        data_shifted = DataUnits(
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

def rb_plot(folder, routine, qubit, format):
    """ """
    from scipy.optimize import curve_fit

    from qibocal.calibrations.protocols.experiments import Experiment

    # Define the function for the fitting process.
    def exp_func(x, A, f, B):
        """ """
        return A * f**x + B

    # Load the data into DataUnits.object.
    data_circs = Data.load_data(folder, routine, "pickle", "circuits")
    data_probs = Data.load_data(folder, routine, "pickle", "probabilities1")
    data_samples = Data.load_data(folder, routine, "pickle", "samples1")
    # Build an Experiment object out of it.
    experiment = Experiment.retrieve_from_dataobjects(
        data_circs, data_samples, data_probs
    )
    # Get the fitting parameters.

    data_fit = Data.load_data(folder, routine, "pickle", f"fitrb").df
    popt = data_fit["A"].values[0], data_fit["f"].values[0], data_fit["B"].values[0]

    # The xaxis is defined by the sequence lengths of the applied circuits.
    xdata = experiment.sequence_lengths
    if experiment.inverse:
        # For each run calculate the probabilities.
        ydata_spread = experiment.probabilities(averaged=False)[:, :, 0].flatten()
        xdata_spread = np.tile(xdata, experiment.runs)
        # The yaxis shows the survival probability, short pm.
        ydata = experiment.probabilities(averaged=True)
        # The ground state probability is used as survival probability.
        pm = np.array(ydata)[:, 0]
    else:
        ydata_spread = experiment.filter_single_qubit(averaged=False)
        xdata_spread = np.tile(xdata, experiment.runs)
        pm = np.average(ydata_spread, axis=0)
        ydata_spread = ydata_spread.flatten()
    # The variance of the variables in 'popt' are calculated with 'pcov'.
    # perr = np.sqrt(np.diag(pcov))
    # Plot the data and the fit.
    x_fit = np.linspace(np.sort(xdata)[0], np.sort(xdata)[-1], num=100)
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
        subplot_titles=(f"Randomized benchmarking, inverse: {experiment.inverse}",),
    )
    c1 = "#6597aa"
    fig.add_trace(
        go.Scatter(
            x=xdata_spread,
            y=ydata_spread,
            line=dict(color=c1),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="",
        ),
        row=1,
        col=1,
    )
    c2 = "#aa6464"
    fig.add_trace(
        go.Scatter(
            x=xdata,
            y=pm,
            line=dict(color=c2),
            mode="markers",
            name="pm",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=exp_func(x_fit, *popt),
            name="A: {:f}, f: {:f}, B: {:f}".format(popt[0], popt[1], popt[2]),
            line=go.scatter.Line(dash="dot"),
        ),
        row=1,
        col=1,
    )
    data = Data.load_data(folder, routine, "pickle", "effectivedepol")
    depol = data.df.to_numpy()[0, 0]
    fig.add_annotation(
        dict(
            font=dict(color="black", size=12),
            x=0,
            y=-0.20,
            showarrow=False,
            text=f"Effective depol param: {depol}",
            textangle=0,
            xanchor="left",
            xref="paper",
            yref="paper",
        )
    )

    return fig


def rb_statistics(folder, routine, qubit, format):
    """ """
    from scipy.optimize import curve_fit

    from qibocal.calibrations.protocols.experiments import Experiment

    # Load the data into DataUnits.object.
    data_circs = Data.load_data(folder, routine, "pickle", "circuits")
    data_probs = Data.load_data(folder, routine, "pickle", "probabilities1")
    data_samples = Data.load_data(folder, routine, "pickle", "samples1")
    # Build an Experiment object out of it.
    experiment = Experiment.retrieve_from_dataobjects(
        data_circs, data_samples, data_probs
    )
    file_exists, count_iterations = True, 0
    data_list = []
    while file_exists:
        count_iterations += 1
        try:
            data_fits = Data.load_data(
                folder, routine, "pickle", f"fits_crossvalidation{count_iterations}"
            )
            data_list.append(data_fits)
        except FileNotFoundError:
            file_exists = False
    fig = make_subplots(
        rows=count_iterations,
        cols=2,
        horizontal_spacing=1,
        vertical_spacing=1,
        subplot_titles=(
            f"Randomized benchmarking Cross Validation, inverse: {experiment.inverse}",
        ),
    )

    c1 = "#6597aa"
    # c2 = "#aa6464"

    for count in range(1, count_iterations):
        fitparams = np.array(data_fits.df.values.tolist()).flatten()
        data_depol = Data.load_data(folder, routine, "pickle", f"effectivedepol{count}")
        depol = data_depol.df.to_numpy()[0, 0]
        fig.add_trace(
            go.Scatter(
                x=fitparams,
                y=np.zeros(len(fitparams)),
                line=dict(color=c1),
                mode="markers",
                marker={"opacity": 0.2, "symbol": "diamond"},
                name=f"effective depol:{depol}",
            ),
            row=count,
            col=1,
        )

        fig.add_trace(
            go.Histogram(x=fitparams),
            row=count,
            col=2,
        )

    return fig
