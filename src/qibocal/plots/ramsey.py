import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import ramsey
from qibocal.plots.utils import get_color, get_data_subfolders, load_data


# Ramsey oscillations
def time_msr_prob(folder, routine, qubit, format):
    figures = []

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""

    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"wait": "ns", "t_max": "ns"},
                options=["qubit", "iteration", "probability"],
            )
        try:
            data_fit = load_data(folder, subfolder, routine, format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data(
                quantities=[
                    "T2",
                    "drive_frequency",
                    "delta_frequency",
                    "popt0",
                    "popt1",
                    "popt2",
                    "popt3",
                    "popt4",
                    "qubit",
                ]
            )

        iterations = data.df["iteration"].unique()
        waits = data.df["wait"].unique()
        data.df = data.df.drop(columns=["qubit"])

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["wait"],
                    y=iteration_data["MSR"] * 1e6,
                    marker_color=get_color(report_n),
                    opacity=opacity,
                    name=f"q{qubit}/r{report_n}",
                    showlegend=not bool(iteration),
                    legendgroup=f"q{qubit}/r{report_n}",
                ),
                row=1,
                col=1,
            )
            # add phase
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["wait"],
                    y=iteration_data["phase"],
                    marker_color=get_color(report_n),
                    opacity=opacity,
                    name=f"q{qubit}/r{report_n}",
                    showlegend=False,
                    legendgroup=f"q{qubit}/r{report_n}",
                ),
                row=1,
                col=2,
            )

        if len(iterations) > 1:
            data.df = data.df.drop(columns=["iteration"])  # pylint: disable=E1101
            fig.add_trace(
                go.Scatter(
                    x=waits,
                    y=data.df.groupby("wait")["MSR"].mean()  # pylint: disable=E1101
                    * 1e6,
                    marker_color=get_color(report_n),
                    name=f"q{qubit}/r{report_n}: Average",
                    showlegend=True,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=1,
            )

        # add fitting trace
        if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
            waitrange = np.linspace(
                min(data.df["wait"]),
                max(data.df["wait"]),
                2 * len(data),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]

            fig.add_trace(
                go.Scatter(
                    x=waitrange,
                    y=ramsey(
                        waitrange,
                        float(data_fit.df["popt0"]),
                        float(data_fit.df["popt1"]),
                        float(data_fit.df["popt2"]),
                        float(data_fit.df["popt3"]),
                        float(data_fit.df["popt4"]),
                    ),
                    name=f"q{qubit}/r{report_n} Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=1,
                col=1,
            )

            fitting_report = (
                fitting_report
                + (
                    f"q{qubit}/r{report_n} | delta_frequency: {params['delta_frequency']:,.0f} Hz<br>"
                )
                + (
                    f"q{qubit}/r{report_n} | drive_frequency: {params['drive_frequency']:,.0f} Hz<br>"
                )
                + (f"q{qubit}/r{report_n} | T2: {params['T2']:,.0f} ns.<br><br>")
            )
        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    # Plot the probability on a new figure
    def iq_to_probability(i, q, mean_gnd, mean_exc):
        state = i + 1j * q
        state = state - mean_gnd
        mean_exc = mean_exc - mean_gnd
        state = state * np.exp(-1j * np.angle(mean_exc))
        mean_exc = mean_exc * np.exp(-1j * np.angle(mean_exc))
        return np.real(state) / np.real(mean_exc)

    runcard = "/home/users/maxime.hantute/qibolab/src/qibolab/runcards/iqm5q.yml"
    import yaml

    with open(runcard) as f:
        runcard = yaml.safe_load(f)
    mean_gnd = complex(
        runcard["characterization"]["single_qubit"][qubit]["mean_gnd_states"]
    )
    mean_exc = complex(
        runcard["characterization"]["single_qubit"][qubit]["mean_exc_states"]
    )

    fig = go.Figure()
    for iteration in iterations:
        iteration_data = data.df[data.df["iteration"] == iteration]
        fig.add_trace(
            go.Scatter(
                x=iteration_data["wait"],
                y=iq_to_probability(
                    iteration_data["i"].to_numpy(),
                    iteration_data["q"].to_numpy(),
                    mean_gnd,
                    mean_exc,
                ),
                marker_color=get_color(report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}",
                showlegend=not bool(iteration),
                legendgroup=f"q{qubit}/r{report_n}",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="Probability",
    )

    figures.append(fig)

    return figures, fitting_report


# Ramsey oscillations
def time_msr(folder, routine, qubit, format):
    figures = []

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""

    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"wait": "ns", "t_max": "ns"},
                options=["qubit", "iteration"],
            )
        try:
            data_fit = load_data(folder, subfolder, routine, format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data(
                quantities=[
                    "T2",
                    "drive_frequency",
                    "delta_frequency",
                    "popt0",
                    "popt1",
                    "popt2",
                    "popt3",
                    "popt4",
                    "qubit",
                ]
            )

        iterations = data.df["iteration"].unique()
        waits = data.df["wait"].unique()
        data.df = data.df.drop(columns=["i", "q", "phase", "qubit"])

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["wait"],
                    y=iteration_data["MSR"] * 1e6,
                    marker_color=get_color(report_n),
                    opacity=opacity,
                    name=f"q{qubit}/r{report_n}",
                    showlegend=not bool(iteration),
                    legendgroup=f"q{qubit}/r{report_n}",
                ),
                row=1,
                col=1,
            )

        if len(iterations) > 1:
            data.df = data.df.drop(columns=["iteration"])  # pylint: disable=E1101
            fig.add_trace(
                go.Scatter(
                    x=waits,
                    y=data.df.groupby("wait")["MSR"].mean()  # pylint: disable=E1101
                    * 1e6,
                    marker_color=get_color(report_n),
                    name=f"q{qubit}/r{report_n}: Average",
                    showlegend=True,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=1,
            )

        # add fitting trace
        if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
            waitrange = np.linspace(
                min(data.df["wait"]),
                max(data.df["wait"]),
                2 * len(data),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]

            fig.add_trace(
                go.Scatter(
                    x=waitrange,
                    y=ramsey(
                        waitrange,
                        float(data_fit.df["popt0"]),
                        float(data_fit.df["popt1"]),
                        float(data_fit.df["popt2"]),
                        float(data_fit.df["popt3"]),
                        float(data_fit.df["popt4"]),
                    ),
                    name=f"q{qubit}/r{report_n} Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=1,
                col=1,
            )

            fitting_report = (
                fitting_report
                + (
                    f"q{qubit}/r{report_n} | delta_frequency: {params['delta_frequency']:,.0f} Hz<br>"
                )
                + (
                    f"q{qubit}/r{report_n} | drive_frequency: {params['drive_frequency']:,.0f} Hz<br>"
                )
                + (f"q{qubit}/r{report_n} | T2: {params['T2']:,.0f} ns.<br><br>")
            )
        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report
