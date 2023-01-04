import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import exp
from qibocal.plots.utils import get_data_subfolders


# Spin echos
def spin_echo_time_msr_phase(folder, routine, qubit, format):

    """Spin echo plotting routine:
    The routine plots the results of a modified Ramsey sequence with an additional Rx(pi) pulse placed symmetrically between the two Rx(pi/2) pulses.
    An exponential fit to this data gives a spin echo decay time T2.
    Args:
        folder (string): Folder name where the data and fitted data is located
        routine (string): Name of the calibration routine that calls this plotting method
        qubit (int): Target qubit to characterize
        format (string): Data file format. Supported formats are .csv and .pkl
    """

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

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    i = 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = DataUnits(quantities={"time": "ns"}, options=["qubit"])

        try:
            data_fit = Data.load_data(
                folder, subfolder, routine, format, f"fit_q{qubit}"
            )
        except:
            data_fit = Data(
                quantities=[
                    "popt0",
                    "popt1",
                    "popt2",
                    "label1",
                ]
            )

        fig.add_trace(
            go.Scatter(
                x=data.get_values("time", "ns"),
                y=data.get_values("MSR", "uV"),
                name=f"q{qubit}/r{i}: spin echo",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.get_values("time", "ns"),
                y=data.get_values("phase", "rad"),
                name=f"q{qubit}/r{i}: spin echo",
            ),
            row=1,
            col=2,
        )

        # add fitting trace
        if len(data) > 0 and len(data_fit) > 0:
            timerange = np.linspace(
                min(data.get_values("time", "ns")),
                max(data.get_values("time", "ns")),
                2 * len(data),
            )
            params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
            fig.add_trace(
                go.Scatter(
                    x=timerange,
                    y=exp(
                        timerange,
                        data_fit.df["popt0"][0],
                        data_fit.df["popt1"][0],
                        data_fit.df["popt2"][0],
                    ),
                    name=f"Fit q{qubit}/r{i}:",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fig.add_annotation(
                dict(
                    font=dict(color="black", size=12),
                    x=i * 0.09,
                    y=-0.25,
                    showarrow=False,
                    text=f"q{qubit}/r{i}: {params[0]}: {data_fit.df[params[0]][0]:.1f} ns.",
                    textangle=0,
                    xanchor="left",
                    xref="paper",
                    yref="paper",
                )
            )
        i += 1

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="time (ns)",
        yaxis2_title="Phase (rad)",
    )
    return fig
