import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import ramsey
from qibocal.plots.utils import get_data_subfolders


# Ramsey oscillations
def time_msr(folder, routine, qubit, format):

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
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
            data = DataUnits(
                name="data",
                quantities={"wait": "ns", "t_max": "ns"},
                options=["qubit"],
            )
        try:
            data_fit = Data.load_data(
                folder, subfolder, routine, format, f"fit_q{qubit}"
            )
        except:
            data_fit = DataUnits()

        datasets = []
        copy = data.df.copy()
        for j in range(len(copy)):
            datasets.append(copy.drop_duplicates("wait"))
            copy.drop(datasets[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["wait"].pint.to("ns").pint.magnitude,
                    y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                    marker_color="rgb(100, 0, 255)",
                    opacity=0.3,
                    name=f"Time q{qubit}/r{i}",
                    showlegend=not bool(j),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=data.df.wait.drop_duplicates()  # pylint: disable=E1101
                .pint.to("ns")
                .pint.magnitude,
                y=data.df.groupby("wait")["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                name=f"Avg time q{qubit}/r{i}",
                marker_color="rgb(100, 0, 255)",
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
                    name=f"Fit q{qubit}/r{i}",
                    line=go.scatter.Line(dash="dot"),
                    marker_color="rgb(255, 130, 67)",
                ),
                row=1,
                col=1,
            )

            fitting_report = fitting_report + (
                f"q{qubit}/r{i} {params[1]}: {data_fit.df[params[1]][0]:.3f} Hz.<br>q{qubit}/r{i} {params[0]}: {data_fit.df[params[0]][0]:.1f} ns<br>q{qubit}/r{i} {params[2]}: {data_fit.df[params[2]][0]:.3f} Hz<br><br>"
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

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )
    return fig
