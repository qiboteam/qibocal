import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import cos
from qibocal.plots.utils import get_data_subfolders


# allXY
def prob_gate(folder, routine, qubit, format):

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    i = 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(
                folder, subfolder, routine, format, f"data_q{qubit}"
            )
        except:
            data = DataUnits(
                quantities={
                    "probability": "dimensionless",
                    "gateNumber": "dimensionless",
                }
            )

        fig.add_trace(
            go.Scatter(
                x=data.get_values("gateNumber", "dimensionless"),
                y=data.get_values("probability", "dimensionless"),
                mode="markers",
                name=f"Probabilities q{qubit}/r{i}",
            ),
            row=1,
            col=1,
        )
        i += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gate sequence number",
        yaxis_title="Z projection probability of qubit state |o>",
    )
    return fig


# allXY iteration
def prob_gate_iteration(folder, routine, qubit, format):

    import numpy as np

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    i = 0
    for subfolder in subfolders:

        try:
            data = DataUnits.load_data(
                folder, subfolder, routine, format, f"data_q{qubit}"
            )
        except:
            data = DataUnits(
                quantities={
                    "probability": "dimensionless",
                    "gateNumber": "dimensionless",
                    "beta_param": "dimensionless",
                }
            )

        gates = len(data.get_values("gateNumber", "dimensionless"))
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
                    name=f"q{qubit}/r{i}: beta_param = {beta_param}",
                    marker_size=16,
                ),
                row=1,
                col=1,
            )
        i += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gate sequence number",
        yaxis_title="Z projection probability of qubit state |o>",
    )
    return fig


# Beta param tuning
def msr_beta(folder, routine, qubit, format):

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
        subplot_titles=(f"beta_param_tuning",),
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
            data = DataUnits(
                name=f"data_q{qubit}", quantities={"beta_param": "dimensionless"}
            )
        try:
            data_fit = DataUnits.load_data(
                folder, subfolder, routine, format, f"fit_q{qubit}"
            )
        except:
            data_fit = DataUnits()

        # c = "#6597aa"
        fig.add_trace(
            go.Scatter(
                x=data.get_values("beta_param", "dimensionless"),
                y=data.get_values("MSR", "uV"),
                # line=dict(color=c),
                mode="markers",
                name=f"q{qubit}/r{i}: [Rx(pi/2) - Ry(pi)] - [Ry(pi/2) - Rx(pi)]",
            ),
            row=1,
            col=1,
        )
        # add fitting traces
        if len(data) > 0 and len(data_fit) > 0:
            beta_param = np.linspace(
                min(data.get_values("beta_param", "dimensionless")),
                max(data.get_values("beta_param", "dimensionless")),
                2 * len(data),
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
                    name=f"Fit q{qubit}/r{i}",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fitting_report = fitting_report + (
                f"q{qubit}/r{i} {params[0]}: {data_fit.df[params[0]][0]:.4f}<br><br>"
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
        xaxis_title="Beta parameter",
        yaxis_title="MSR[uV]",
    )
    return fig
