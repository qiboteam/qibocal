import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import cos
from qibocal.plots.utils import get_color, get_data_subfolders

# allXY rotations
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


# allXY
def allXY(folder, routine, qubit, format):
    figures = []

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
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = Data(
                name="data",
                quantities={"probability", "gateNumber", "qubit", "iteration"},
            )
        iterations = data.df["iteration"].unique()
        gate_numbers = data.df["gateNumber"].unique()

        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["gateNumber"],
                    y=iteration_data["probability"],
                    marker_color=get_color(report_n),
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
                x=gate_numbers,
                y=data.df.groupby("gateNumber", as_index=False)[
                    "probability"
                ].mean(),  # pylint: disable=E1101
                name=f"q{qubit}/r{report_n}: Average Probability",
                marker_color=get_color(report_n),
                mode="markers",
                text=gatelist,
                textposition="bottom center",
            ),
            row=1,
            col=1,
        )
        report_n += 1

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

    fig.add_hline(
        y=-1,
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
        yaxis_title="Expectation value of Z",
    )

    figures.append(fig)

    return figures


# allXY
def allXY_drag_pulse_tuning(folder, routine, qubit, format):
    figures = []

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
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = Data(
                quantities={
                    "probability",
                    "gateNumber",
                    "beta_param",
                    "qubit",
                    "iteration",
                },
            )

        iterations = data.df["iteration"].unique()
        beta_params = data.df["beta_param"].unique()
        gate_numbers = data.df["gateNumber"].unique()

        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            for j, beta_param in enumerate(beta_params):
                beta_param_data = iteration_data[
                    iteration_data["beta_param"] == beta_param
                ]
                fig.add_trace(
                    go.Scatter(
                        x=beta_param_data["gateNumber"],
                        y=beta_param_data["probability"],
                        marker_color=get_color(report_n * len(beta_params) + j),
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

    fig.add_hline(
        y=-1,
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
        yaxis_title="Expectation value of Z",
    )

    figures.append(fig)

    return figures


# beta param tuning
def drag_pulse_tuning(folder, routine, qubit, format):
    figures = []

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
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name="data",
                quantities={"beta_param": "dimensionless"},
                options=["qubit", "iteration"],
            )
        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data()

        iterations = data.df["iteration"].unique()
        beta_params = data.df["beta_param"].pint.magnitude.unique()

        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["beta_param"].pint.magnitude,
                    y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                    marker_color=get_color(report_n),
                    mode="markers",
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
                x=beta_params,
                y=data.df.groupby("beta_param", as_index=False)
                .mean()["MSR"]  # pylint: disable=E1101
                .pint.to("uV")
                .pint.magnitude,
                name=f"q{qubit}/r{report_n}: Average MSR",
                marker_color=get_color(report_n),
                mode="markers",
            ),
            row=1,
            col=1,
        )

        # add fitting traces

        if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
            beta_range = np.linspace(
                min(data.get_values("beta_param", "dimensionless")),
                max(data.get_values("beta_param", "dimensionless")),
                20,
            )

            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]
            fig.add_trace(
                go.Scatter(
                    x=beta_range,
                    y=cos(
                        beta_range,
                        data_fit.get_values("popt0"),
                        data_fit.get_values("popt1"),
                        data_fit.get_values("popt2"),
                        data_fit.get_values("popt3"),
                    ),
                    name=f"q{qubit}/r{report_n}: Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=1,
                col=1,
            )
            fitting_report = fitting_report + (
                f"q{qubit}/r{report_n} optimal_beta_param: {params['optimal_beta_param']:.4f}<br><br>"
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

    figures.append(fig)

    return figures
