import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import cos


def allXY(folder, routine, qubit, format):

    try:
        data = Data.load_data(folder, routine, format, "data")
        data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
    except:
        data = Data(
            name="data",
            quantities={"probability", "gateNumber", "qubit"},
        )

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY",),
    )

    datasets = []
    copy = data.df.copy()
    for i in range(len(copy)):
        datasets.append(copy.drop_duplicates("gateNumber"))
        copy.drop(datasets[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets[-1]["gateNumber"],
                y=datasets[-1]["probability"],
                marker_color="rgb(100, 0, 255)",
                mode="markers",
                opacity=0.3,
                name="Probability",
                showlegend=not bool(i),
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
            name="Average Probability",
            marker_color="rgb(100, 0, 255)",
            mode="markers",
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
        data = DataUnits.load_data(folder, routine, format, "data")
        data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
    except:
        data = DataUnits(
            quantities={
                "probability": "dimensionless",
                "gateNumber": "dimensionless",
                "beta_param": "dimensionless",
            },
            options=["qubit"],
        )

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY",),
    )

    copy = data.df.copy()

    beta_params = (
        copy.drop_duplicates("beta_param")["beta_param"]
        .pint.to("dimensionless")
        .pint.magnitude
    )
    gateNumber = (
        copy.drop_duplicates("gateNumber")["gateNumber"]
        .pint.to("dimensionless")
        .pint.magnitude
    )
    size = len(copy.drop_duplicates("gateNumber")) * len(
        copy.drop_duplicates("beta_param")
    )
    software_averages = len(copy) // size

    for i in range(software_averages):
        test = data.df[size * i : size * i + size]

        for j, beta_param in enumerate(beta_params):

            fig.add_trace(
                go.Scatter(
                    x=gateNumber,
                    y=test[test["beta_param"] == beta_param]["probability"]
                    .pint.to("dimensionless")
                    .pint.magnitude,
                    marker_color="#" + "{:06x}".format(j * 99999),
                    mode="markers+lines",
                    opacity=0.5,
                    name=f"beta {beta_param}",
                    showlegend=not bool(i),
                    legendgroup=f"group{j}",
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
        data = DataUnits.load_data(folder, routine, format, "data")
        data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
    except:
        data = DataUnits(
            name="data", quantities={"beta_param": "dimensionless"}, options=["qubit"]
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

    datasets = []
    copy = data.df.copy()
    for i in range(len(copy)):
        datasets.append(copy.drop_duplicates("beta_param"))
        copy.drop(datasets[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets[-1]["beta_param"].pint.to("dimensionless").pint.magnitude,
                y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                marker_color="rgb(100, 0, 255)",
                mode="markers",
                opacity=0.3,
                name="Probability",
                showlegend=not bool(i),
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
            name="Average MSR",
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
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
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
