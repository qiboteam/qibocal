import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import exp
from qibocal.plots.utils import get_data_subfolders


# T1
def t1_time_msr_phase(folder, routine, qubit, format):
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
    fitting_report = ""
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
            data_fit = DataUnits()

        datasets = []
        copy = data.df.copy()
        for i in range(len(copy)):
            datasets.append(copy.drop_duplicates("time"))
            copy.drop(datasets[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["time"].pint.to("ns").pint.magnitude,
                    y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                    marker_color="rgb(100, 0, 255)",
                    opacity=0.3,
                    name=f"T1 q{qubit}/r{i}",
                    showlegend=not bool(i),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["time"].pint.to("ns").pint.magnitude,
                    y=datasets[-1]["phase"].pint.to("rad").pint.magnitude,
                    marker_color="rgb(102, 180, 71)",
                    name=f"T1 q{qubit}/r{i}",
                    opacity=0.3,
                    showlegend=not bool(i),
                    legendgroup="group2",
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=data.df.time.drop_duplicates()  # pylint: disable=E1101
                .pint.to("ns")
                .pint.magnitude,
                y=data.df.groupby("time")["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                name=f"avg T1 q{qubit}/r{i}",
                marker_color="rgb(100, 0, 255)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.df.time.drop_duplicates()  # pylint: disable=E1101
                .pint.to("ns")
                .pint.magnitude,
                y=data.df.groupby("time")["phase"]  # pylint: disable=E1101
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                name=f"avg T1 q{qubit}/r{i}",
                marker_color="rgb(204, 102, 102)",
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
                        data_fit.get_values("popt0"),
                        data_fit.get_values("popt1"),
                        data_fit.get_values("popt2"),
                    ),
                    name=f"Fit q{qubit}/r{i}",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fitting_report = fitting_report + (
                f"q{qubit}/r{i} {params[0]}: {data_fit.df[params[0]][0]:.1f} ns.<br><br>"
            )

        i += 1

    fig.add_annotation(
        dict(
            font=dict(color="black", size=12),
            x=0,
            y=1.2,
            showarrow=False,
            text="<b>FITTING DATA</b>",
            font_family="Arial",
            font_size=20,
            textangle=0,
            xanchor="left",
            xref="paper",
            yref="paper",
            font_color="#5e9af1",
            hovertext=fitting_report,
        )
    )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="time (ns)",
        yaxis2_title="Phase (rad)",
    )
    return fig
