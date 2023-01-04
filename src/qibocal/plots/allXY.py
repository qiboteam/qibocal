import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import cos
from qibocal.plots.utils import get_data_subfolders

gatelist = [
    ["I", "I"],
    ["RX(pi)", "RX(pi)"],
    ["RY(pi)", "RY(pi)"],
    ["RX(pi)", "RY(pi)"],
    ["RY(pi)", "RX(pi)"],
    ["RX(pi/2)", "I"],
    ["RY(pi/2)", "I"],
    ["RX(pi/2)", "RY(pi/2)"],
    ["RY(pi/2)", "RX(pi/2)"],
    ["RX(pi/2)", "RY(pi)"],
    ["RY(pi/2)", "RX(pi)"],
    ["RX(pi)", "RY(pi/2)"],
    ["RY(pi)", "RX(pi/2)"],
    ["RX(pi/2)", "RX(pi)"],
    ["RX(pi)", "RX(pi/2)"],
    ["RY(pi/2)", "RY(pi)"],
    ["RY(pi)", "RY(pi/2)"],
    ["RX(pi)", "I"],
    ["RY(pi)", "I"],
    ["RX(pi/2)", "RX(pi/2)"],
    ["RY(pi/2)", "RY(pi/2)"],
]


def allXY(folder, routine, qubit, format):

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
            data = Data.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = Data(
                name="data",
                quantities={"probability", "gateNumber", "qubit"},
            )

        datasets = []
        copy = data.df.copy()
        for j in range(len(copy)):
            datasets.append(copy.drop_duplicates("gateNumber"))
            copy.drop(datasets[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["gateNumber"],
                    y=datasets[-1]["probability"],
                    marker_color="rgb(100, 0, 255)",
                    mode="markers",
                    text=gatelist,
                    textposition="bottom center",
                    opacity=0.3,
                    name=f"Probabilities q{qubit}/r{i}",
                    showlegend=not bool(j),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=data.df.gateNumber.drop_duplicates(),  # pylint: disable=E1101
                y=data.df.groupby("gateNumber")[
                    "probability"
                ].mean(),  # pylint: disable=E1101
                name=f"Average Probabilities q{qubit}/r{i}",
                marker_color="rgb(100, 0, 255)",
                mode="markers",
                text=gatelist,
                textposition="bottom center",
            ),
            row=1,
            col=1,
        )

        fig.add_hline(
            y=-1,
            line_width=2,
            line_dash="dash",
            line_color="grey",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=0,
            line_width=2,
            line_dash="dash",
            line_color="grey",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=1,
            line_width=2,
            line_dash="dash",
            line_color="grey",
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
            data = Data.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = Data(quantities=["probability" "gateNumber" "beta_param", "qubit"])

        copy = data.df.copy()

        beta_params = copy.drop_duplicates("beta_param")["beta_param"]
        gateNumber = copy.drop_duplicates("gateNumber")["gateNumber"]
        size = len(copy.drop_duplicates("gateNumber")) * len(
            copy.drop_duplicates("beta_param")
        )

        if size > 0:
            software_averages = len(copy) // size

            for k in range(software_averages):
                test = data.df[size * i : size * i + size]
                for j, beta_param in enumerate(beta_params):
                    fig.add_trace(
                        go.Scatter(
                            x=gateNumber,
                            y=test[test["beta_param"] == beta_param]["probability"],
                            marker_color="#" + "{:06x}".format(j * 99999),
                            mode="markers+lines",
                            opacity=0.5,
                            name=f"q{qubit}/r{i}: beta_param = {beta_param}",
                            showlegend=not bool(k),
                            legendgroup=f"group{j}",
                            text=gatelist,
                            textposition="bottom center",
                        ),
                        row=1,
                        col=1,
                    )

        fig.add_hline(
            y=-1,
            line_width=2,
            line_dash="dash",
            line_color="grey",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=0,
            line_width=2,
            line_dash="dash",
            line_color="grey",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=1,
            line_width=2,
            line_dash="dash",
            line_color="grey",
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
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = DataUnits(
                name="data",
                quantities={"beta_param": "dimensionless"},
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
            datasets.append(copy.drop_duplicates("beta_param"))
            copy.drop(datasets[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["beta_param"]
                    .pint.to("dimensionless")
                    .pint.magnitude,
                    y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                    marker_color="rgb(100, 0, 255)",
                    mode="markers",
                    opacity=0.3,
                    name=f"q{qubit}/r{i}: [Rx(pi/2) - Ry(pi)] - [Ry(pi/2) - Rx(pi)]",
                    showlegend=not bool(j),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=data.df.beta_param.drop_duplicates().pint.magnitude,  # pylint: disable=E1101
                y=data.df.groupby("beta_param")["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                name=f"q{qubit}/r{i}: avg [Rx(pi/2) - Ry(pi)] - [Ry(pi/2) - Rx(pi)]",
                marker_color="rgb(100, 0, 255)",
                mode="markers",
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
                    name=f"Fit q{qubit}/r{i}",
                    line=go.scatter.Line(dash="dot"),
                    marker_color="rgb(255, 130, 67)",
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
