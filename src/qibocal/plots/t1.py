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
            data = DataUnits.load_data(
                folder, subfolder, routine, format, f"data_q{qubit}"
            )
        except:
            data = DataUnits(quantities={"Time": "ns"})

        try:
            data_fit = Data.load_data(
                folder, subfolder, routine, format, f"fit_q{qubit}"
            )
        except:
            data_fit = DataUnits()

        fig.add_trace(
            go.Scatter(
                x=data.get_values("Time", "ns"),
                y=data.get_values("MSR", "uV"),
                name=f"T1 q{qubit}/r{i}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.get_values("Time", "ns"),
                y=data.get_values("phase", "rad"),
                name=f"T1 q{qubit}/r{i}",
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
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Time (ns)",
        yaxis2_title="Phase (rad)",
    )
    return fig
