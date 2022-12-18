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


from colorsys import hls_to_rgb


def _get_color(number):
    return "rgb" + str(hls_to_rgb((0.75 - number * 3 / 20) % 1, 0.4, 0.75))


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
    report_n = 0
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
        software_averages = len(copy)
        for iteration in range(software_averages):
            datasets.append(copy.drop_duplicates("gateNumber"))
            copy.drop(datasets[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["gateNumber"],
                    y=datasets[-1]["probability"],
                    marker_color=_get_color(report_n),
                    mode="markers",
                    text=gatelist,
                    textposition="bottom center",
                    opacity=0.3,
                    name=f"q{qubit}/r{report_n}: Probability",
                    showlegend=not bool(iteration),
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
                name=f"q{qubit}/r{report_n}: Average Probability",
                marker_color=_get_color(report_n),
                mode="markers",
                text=gatelist,
                textposition="bottom center",
            ),
            row=1,
            col=1,
        )
        report_n += 1

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
def allXY_drag_pulse_tuning(folder, routine, qubit, format):

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    for subfolder in subfolders:

        try:
            data = Data.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)
        except:
            data = Data(
                quantities={
                    "probability",
                    "gateNumber",
                    "beta_param",
                    "qubit",
                },
            )

        beta_params = data.df.drop_duplicates("beta_param")["beta_param"].to_numpy()

        total_size = len(data.df)
        software_averages = (
            total_size // (21 * len(beta_params))
        ) + 1  # num software averages (last incomplete)
        software_average_size = 21 * len(beta_params)

        for iteration in range(software_averages):
            iteration_data = data.df[
                software_average_size
                * iteration : min(
                    total_size,
                    software_average_size * iteration + software_average_size,
                )
            ]
            for j, beta_param in enumerate(beta_params):
                beta_param_data = iteration_data[
                    iteration_data["beta_param"] == beta_param
                ]

                fig.add_trace(
                    go.Scatter(
                        x=beta_param_data["gateNumber"].to_numpy(),
                        y=beta_param_data["probability"].to_numpy(),
                        marker_color=_get_color(report_n * len(beta_params) + j),
                        mode="markers+lines",
                        opacity=0.5,
                        name=f"q{qubit}/r{report_n}: beta {beta_param}",
                        showlegend=not bool(iteration),
                        legendgroup=f"group{j}",
                        text=gatelist,
                        textposition="bottom center",
                    ),
                    row=1,
                    col=1,
                )
        report_n += 1

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


# beta param tuning
def msr_beta(folder, routine, qubit, format):

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
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
            data_fit = Data.load_data(folder, subfolder, routine, format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == int(qubit)].reset_index(
                drop=True
            )
        except:
            data_fit = Data()

        datasets = []
        copy = data.df.copy()
        for i in range(len(copy)):
            datasets.append(copy.drop_duplicates("beta_param"))
            copy.drop(datasets[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["beta_param"]
                    .pint.to("dimensionless")
                    .pint.magnitude,
                    y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                    marker_color=_get_color(report_n),
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
                name=f"Average MSR q{qubit}/r{report_n}",
                marker_color=_get_color(report_n),
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
                    name=f"Fit q{qubit}/r{report_n}",
                    line=go.scatter.Line(dash="dot"),
                    marker_color="rgb(255, 130, 67)",
                ),
                row=1,
                col=1,
            )
            params.remove("qubit")
            fitting_report = fitting_report + (
                f"q{qubit}/r{report_n} {params[0]}: {data_fit.df[params[0]][0]:.4f}<br><br>"
            )

            report_n += 1

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
        yaxis_title="MSR[uV] [Rx(pi/2) - Ry(pi)] - [Ry(pi/2) - Rx(pi)]",
    )
    return fig
