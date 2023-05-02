import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.plots.utils import get_color_state0, get_color_state1, get_data_subfolders


# Plot RO optimization with frequency
def ro_frequency(folder, routine, qubit, format):
    fig = go.Figure()

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    max_x, max_y, min_x, min_y = 0, 0, 0, 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name="data",
                quantities={"frequency": "Hz", "delta_frequency": "Hz"},
                options=["iteration", "state"],
            )

        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, "fit")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]

        except:
            data_fit = Data(
                name="fit",
                quantities=[
                    "frequency",
                    "delta_frequency",
                    "rotation_angle",
                    "threshold",
                    "fidelity",
                    "assignment_fidelity",
                    "average_state0",
                    "average_state1",
                ],
            )

        # Plot raw results with sliders
        annotations_dict = []
        for frequency in data.df["delta_frequency"].unique():
            state0_data = data.df[
                (data.df["delta_frequency"] == frequency) & (data.df["state"] == 0)
            ]
            state1_data = data.df[
                (data.df["delta_frequency"] == frequency) & (data.df["state"] == 1)
            ]
            fit_data = data_fit.df[
                data_fit.df["delta_frequency"] == frequency.magnitude
            ]

            # print(fit_data)
            fig.add_trace(
                go.Scatter(
                    x=state0_data["i"].pint.to("V").pint.magnitude,
                    y=state0_data["q"].pint.to("V").pint.magnitude,
                    name=f"q{qubit}/r{report_n}: state 0",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3, color=get_color_state0(report_n)),
                    visible=False,
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=state1_data["i"].pint.to("V").pint.magnitude,
                    y=state1_data["q"].pint.to("V").pint.magnitude,
                    name=f"q{qubit}/r{report_n}: state 1",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3, color=get_color_state1(report_n)),
                    visible=False,
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=[complex(fit_data.iloc[0]["average_state0"]).real],
                    y=[complex(fit_data.iloc[0]["average_state0"]).imag],
                    name=f"q{qubit}/r{report_n}: mean state 0",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    showlegend=True,
                    visible=False,
                    mode="markers",
                    marker=dict(size=10, color=get_color_state0(report_n)),
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=[complex(fit_data.iloc[0]["average_state1"]).real],
                    y=[complex(fit_data.iloc[0]["average_state1"]).imag],
                    name=f"avg q{qubit}/r{report_n}: mean state 1",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    showlegend=True,
                    visible=False,
                    mode="markers",
                    marker=dict(size=10, color=get_color_state1(report_n)),
                ),
            )

            min_x = min(min_x, state0_data["i"].min(), state1_data["i"].min())
            min_y = min(min_y, state0_data["q"].min(), state1_data["q"].min())
            max_x = max(max_x, state0_data["i"].max(), state1_data["i"].max())
            max_y = max(max_y, state0_data["q"].max(), state1_data["q"].max())

        report_n += 1
    # Show data for the first frequency
    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.data[2].visible = True
    fig.data[3].visible = True

    # Add slider
    steps = []
    for i, freq in enumerate(data.df["frequency"].unique()):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
            ],
            label=f"{freq:,.0f}",
        )
        for j in range(4):
            step["args"][0]["visible"][i * 4 + j] = True
        steps.append(step)

    sliders = [
        dict(
            currentvalue={"prefix": "frequency: "},
            steps=steps,
        )
    ]

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="i (V)",
        yaxis_title="q (V)",
        xaxis_range=(min_x, max_x),
        yaxis_range=(min_y, max_y),
        sliders=sliders,
        title=f"q{qubit}",
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    # Plot the fidelity as a function of frequency
    fig_fidelity = go.Figure()

    fig_fidelity.add_trace(
        go.Scatter(
            x=data_fit.df["frequency"].astype(int), y=data_fit.df["assignment_fidelity"]
        )
    )
    fig_fidelity.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="delta frequency (Hz)",
        yaxis_title="assignment fidelity",
        title=f"q{qubit}",
    )
    # Add fitting report for the best fidelity
    fit_data = data_fit.df[data_fit.df["fidelity"] == data_fit.df["fidelity"].max()]
    title_text = f"q{qubit}/r{report_n} | frequency: {int(fit_data['frequency'].to_numpy()[0]):,.0f}<br>"
    title_text += f"q{qubit}/r{report_n} | average state 0: ({complex(fit_data['average_state0'].to_numpy()[0]):.6f})<br>"
    title_text += f"q{qubit}/r{report_n} | average state 1: ({complex(fit_data['average_state1'].to_numpy()[0]):.6f})<br>"
    title_text += f"q{qubit}/r{report_n} | rotation angle: {float(fit_data['rotation_angle'].to_numpy()[0]):.3f}<br>"
    title_text += f"q{qubit}/r{report_n} | threshold: {float(fit_data['threshold'].to_numpy()[0]):.6f}<br>"
    title_text += f"q{qubit}/r{report_n} | fidelity: {float(fit_data['fidelity'].to_numpy()[0]):.3f}<br>"
    title_text += f"q{qubit}/r{report_n} | assignment fidelity: {float(fit_data['assignment_fidelity'].to_numpy()[0]):.3f}<br><br>"
    fitting_report = fitting_report + title_text
    return [fig, fig_fidelity], fitting_report


# Plot RO optimization with amplitude
def ro_amplitude(folder, routine, qubit, format):
    fig = go.Figure()

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    max_x, max_y, min_x, min_y = 0, 0, 0, 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name="data",
                quantities={
                    "amplitude": "dBm",
                    "delta_amplitude": "dBm",
                },
                options=["iteration", "state"],
            )

        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, "fit")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]

        except:
            data_fit = Data(
                name="fit",
                quantities=[
                    "amplitude",
                    "delta_amplitude",
                    "rotation_angle",
                    "threshold",
                    "fidelity",
                    "assignment_fidelity",
                    "average_state0",
                    "average_state1",
                ],
            )

        # Plot raw results with sliders
        for amplitude in data.df["delta_amplitude"].unique():
            state0_data = data.df[
                (data.df["delta_amplitude"] == amplitude) & (data.df["state"] == 0)
            ]
            state1_data = data.df[
                (data.df["delta_amplitude"] == amplitude) & (data.df["state"] == 1)
            ]
            fit_data = data_fit.df[
                data_fit.df["delta_amplitude"] == amplitude.magnitude
            ]

            # print(fit_data)
            fig.add_trace(
                go.Scatter(
                    x=state0_data["i"].pint.to("V").pint.magnitude,
                    y=state0_data["q"].pint.to("V").pint.magnitude,
                    name=f"q{qubit}/r{report_n}: state 0",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3, color=get_color_state0(report_n)),
                    visible=False,
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=state1_data["i"].pint.to("V").pint.magnitude,
                    y=state1_data["q"].pint.to("V").pint.magnitude,
                    name=f"q{qubit}/r{report_n}: state 1",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3, color=get_color_state1(report_n)),
                    visible=False,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[complex(fit_data.iloc[0]["average_state0"]).real],
                    y=[complex(fit_data.iloc[0]["average_state0"]).imag],
                    name=f"q{qubit}/r{report_n}: mean state 0",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    showlegend=True,
                    visible=False,
                    mode="markers",
                    marker=dict(size=10, color=get_color_state0(report_n)),
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=[complex(fit_data.iloc[0]["average_state1"]).real],
                    y=[complex(fit_data.iloc[0]["average_state1"]).imag],
                    name=f"avg q{qubit}/r{report_n}: mean state 1",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    showlegend=True,
                    visible=False,
                    mode="markers",
                    marker=dict(size=10, color=get_color_state1(report_n)),
                ),
            )

            min_x = min(min_x, state0_data["i"].min(), state1_data["i"].min())
            min_y = min(min_y, state0_data["q"].min(), state1_data["q"].min())
            max_x = max(max_x, state0_data["i"].max(), state1_data["i"].max())
            max_y = max(max_y, state0_data["q"].max(), state1_data["q"].max())

        report_n += 1
    # Show data for the first amplitude
    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.data[2].visible = True
    fig.data[3].visible = True

    # Add slider
    steps = []
    for i, amp in enumerate(data.df["amplitude"].unique()):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
            ],
            label=f"{amp.magnitude:.4f}",
        )
        for j in range(4):
            step["args"][0]["visible"][i * 4 + j] = True
        steps.append(step)

    sliders = [
        dict(
            currentvalue={"prefix": "amplitude: "},
            steps=steps,
        )
    ]

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="i (V)",
        yaxis_title="q (V)",
        xaxis_range=(min_x, max_x),
        yaxis_range=(min_y, max_y),
        sliders=sliders,
        title=f"q{qubit}",
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    # Plot the fidelity as a function of amplitude
    fig_fidelity = go.Figure()

    fig_fidelity.add_trace(
        go.Scatter(
            x=data_fit.df["delta_amplitude"], y=data_fit.df["assignment_fidelity"]
        )
    )
    fig_fidelity.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="amplitude_factor",
        yaxis_title="assignment_fidelity",
        title=f"q{qubit}",
    )
    # Add fitting report for the best fidelity
    fit_data = data_fit.df[data_fit.df["fidelity"] == data_fit.df["fidelity"].max()]
    title_text = f"q{qubit}/r{report_n} | delta_amplitude: {float(fit_data['delta_amplitude'].to_numpy()[0]):.6f}<br>"
    title_text += f"q{qubit}/r{report_n} | average state 0: ({complex(fit_data['average_state0'].to_numpy()[0]):.6f})<br>"
    title_text += f"q{qubit}/r{report_n} | average state 1: ({complex(fit_data['average_state1'].to_numpy()[0]):.6f})<br>"
    title_text += f"q{qubit}/r{report_n} | rotation angle: {float(fit_data['rotation_angle'].to_numpy()[0]):.3f}<br>"
    title_text += f"q{qubit}/r{report_n} | threshold: {float(fit_data['threshold'].to_numpy()[0]):.6f}<br>"
    title_text += f"q{qubit}/r{report_n} | fidelity: {float(fit_data['fidelity'].to_numpy()[0]):.3f}<br>"
    title_text += f"q{qubit}/r{report_n} | assignment fidelity: {float(fit_data['assignment_fidelity'].to_numpy()[0]):.3f}<br><br>"
    fitting_report = fitting_report + title_text
    return [fig, fig_fidelity], fitting_report


# Plot RO optimization with power
# Plot RO optimization with amplitude
def ro_power(folder, routine, qubit, format):
    fig = go.Figure()

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    max_x, max_y, min_x, min_y = 0, 0, 0, 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name="data",
                quantities={
                    "power": "dBm",
                    "delta_power": "dBm",
                },
                options=["iteration", "state"],
            )

        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, "fit")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]

        except:
            data_fit = Data(
                name="fit",
                quantities=[
                    "power",
                    "delta_power",
                    "rotation_angle",
                    "threshold",
                    "fidelity",
                    "assignment_fidelity",
                    "average_state0",
                    "average_state1",
                ],
            )

        # Plot raw results with sliders
        for power in data.df["delta_power"].unique():
            state0_data = data.df[
                (data.df["delta_power"] == power) & (data.df["state"] == 0)
            ]
            state1_data = data.df[
                (data.df["delta_power"] == power) & (data.df["state"] == 1)
            ]
            fit_data = data_fit.df[data_fit.df["delta_power"] == power.magnitude]

            # print(fit_data)
            fig.add_trace(
                go.Scatter(
                    x=state0_data["i"].pint.to("V").pint.magnitude,
                    y=state0_data["q"].pint.to("V").pint.magnitude,
                    name=f"q{qubit}/r{report_n}: state 0",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3, color=get_color_state0(report_n)),
                    visible=False,
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=state1_data["i"].pint.to("V").pint.magnitude,
                    y=state1_data["q"].pint.to("V").pint.magnitude,
                    name=f"q{qubit}/r{report_n}: state 1",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3, color=get_color_state1(report_n)),
                    visible=False,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[complex(fit_data.iloc[0]["average_state0"]).real],
                    y=[complex(fit_data.iloc[0]["average_state0"]).imag],
                    name=f"q{qubit}/r{report_n}: mean state 0",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    showlegend=True,
                    visible=False,
                    mode="markers",
                    marker=dict(size=10, color=get_color_state0(report_n)),
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=[complex(fit_data.iloc[0]["average_state1"]).real],
                    y=[complex(fit_data.iloc[0]["average_state1"]).imag],
                    name=f"avg q{qubit}/r{report_n}: mean state 1",
                    # legendgroup=f"q{qubit}/r{report_n}",
                    showlegend=True,
                    visible=False,
                    mode="markers",
                    marker=dict(size=10, color=get_color_state1(report_n)),
                ),
            )

            min_x = min(min_x, state0_data["i"].min(), state1_data["i"].min())
            min_y = min(min_y, state0_data["q"].min(), state1_data["q"].min())
            max_x = max(max_x, state0_data["i"].max(), state1_data["i"].max())
            max_y = max(max_y, state0_data["q"].max(), state1_data["q"].max())

        report_n += 1
    # Show data for the first frequency
    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.data[2].visible = True
    fig.data[3].visible = True

    # Add slider
    steps = []
    for i, pow in enumerate(data.df["power"].unique()):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
            ],
            label=f"{pow.magnitude:.4f}",
        )
        for j in range(4):
            step["args"][0]["visible"][i * 4 + j] = True
        steps.append(step)

    sliders = [
        dict(
            currentvalue={"prefix": "power: "},
            steps=steps,
        )
    ]

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="i (V)",
        yaxis_title="q (V)",
        xaxis_range=(min_x, max_x),
        yaxis_range=(min_y, max_y),
        sliders=sliders,
        title=f"q{qubit}",
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    # Plot the fidelity as a function of power
    fig_fidelity = go.Figure()

    fig_fidelity.add_trace(
        go.Scatter(x=data_fit.df["power"], y=data_fit.df["fidelity"])
    )
    fig_fidelity.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="delta power (dBm)",
        yaxis_title="fidelity (ratio)",
        title=f"q{qubit}",
    )
    # Add fitting report for the best fidelity
    fit_data = data_fit.df[data_fit.df["fidelity"] == data_fit.df["fidelity"].max()]
    title_text = f"q{qubit}/r{report_n} | power: {float(fit_data['power'].to_numpy()[0]):.6f}<br>"
    title_text += f"q{qubit}/r{report_n} | average state 0: ({complex(fit_data['average_state0'].to_numpy()[0]):.6f})<br>"
    title_text += f"q{qubit}/r{report_n} | average state 1: ({complex(fit_data['average_state1'].to_numpy()[0]):.6f})<br>"
    title_text += f"q{qubit}/r{report_n} | rotation angle: {float(fit_data['rotation_angle'].to_numpy()[0]):.3f}<br>"
    title_text += f"q{qubit}/r{report_n} | threshold: {float(fit_data['threshold'].to_numpy()[0]):.6f}<br>"
    title_text += f"q{qubit}/r{report_n} | fidelity: {float(fit_data['fidelity'].to_numpy()[0]):.3f}<br>"
    title_text += f"q{qubit}/r{report_n} | assignment fidelity: {float(fit_data['assignment_fidelity'].to_numpy()[0]):.3f}<br><br>"
    fitting_report = fitting_report + title_text
    return [fig, fig_fidelity], fitting_report
