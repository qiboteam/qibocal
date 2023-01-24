import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.plots.utils import get_color_state0, get_color_state1, get_data_subfolders


# For calibrate qubit states
def ro_frequency(folder, routine, qubit, format):

    fig = go.Figure()

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
                quantities={"frequency": "Hz", "delta_frequency": "Hz"},
                options=["iteration", "state"],
            )

        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, "fit")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]

        except:
            data_fit = Data(
                name="fit",
                quantities={"frequency": "Hz", "delta_frequency": "Hz"},
                options=[
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
        for delta_frequency in data.df["delta_frequency"].unique():
            state0_data = data.df[
                (data.df["delta_frequency"] == delta_frequency)
                & (data.df["state"] == 0)
            ]
            state1_data = data.df[
                (data.df["delta_frequency"] == delta_frequency)
                & (data.df["state"] == 1)
            ]
            fit_data = data_fit.df[
                data_fit.df["delta_frequency"] == delta_frequency.magnitude
            ]
            fit_data["average_state0"] = data_fit.df["average_state0"].apply(
                lambda x: complex(x)
            )
            fit_data["average_state1"] = data_fit.df["average_state1"].apply(
                lambda x: complex(x)
            )

            # print(fit_data)
            fig.add_trace(
                go.Scatter(
                    x=state0_data["i"].pint.to("V").pint.magnitude,
                    y=state0_data["q"].pint.to("V").pint.magnitude,
                    name=f"q{qubit}/r{report_n}: state 0",
                    legendgroup=f"q{qubit}/r{report_n}",
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
                    legendgroup=f"q{qubit}/r{report_n}",
                    mode="markers",
                    showlegend=True,
                    opacity=0.7,
                    marker=dict(size=3, color=get_color_state1(report_n)),
                    visible=False,
                ),
            )
            # print([float(fit_data["average_state1"].apply(lambda x: np.real(x)))])
            # print([float(fit_data["average_state0"].apply(lambda x: np.imag(x)))])
            fig.add_trace(
                go.Scatter(
                    x=[float(fit_data["average_state0"].apply(lambda x: np.real(x)))],
                    y=[float(fit_data["average_state0"].apply(lambda x: np.imag(x)))],
                    name=f"q{qubit}/r{report_n}: mean state 0",
                    legendgroup=f"q{qubit}/r{report_n}",
                    showlegend=True,
                    visible=False,
                    mode="markers",
                    marker=dict(size=10, color=get_color_state0(report_n)),
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=[float(fit_data["average_state1"].apply(lambda x: np.real(x)))],
                    y=[float(fit_data["average_state1"].apply(lambda x: np.imag(x)))],
                    name=f"avg q{qubit}/r{report_n}: mean state 1",
                    legendgroup=f"q{qubit}/r{report_n}",
                    showlegend=True,
                    visible=False,
                    mode="markers",
                    marker=dict(size=10, color=get_color_state1(report_n)),
                ),
            )

            # Add fitting report
            title_text = f"q{qubit}/r{report_n}/r{delta_frequency}<br>"
            # title_text += f"average state 0: ({fit_data['average_state0'][0]:.6f})<br>"
            # title_text += f"average state 1: ({fit_data['average_state1'][0]:.6f})<br>"
            # title_text += (
            #     f"rotation angle = {fit_data['rotation_angle'][0]:.3f} / threshold = {fit_data['threshold'][0]:.6f}<br>"
            # )
            # title_text += f"fidelity = {fit_data['fidelity'][0]:.3f} / assignment fidelity = {fit_data['assignment_fidelity'][0]:.3f}<br><br>"
            fitting_report = fitting_report + title_text

            annotations_dict.append(
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
                ),
            )
            fig.add_annotation(annotations_dict[-1], visible=False)
        report_n += 1
    # Show data for the first frequency
    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.data[2].visible = True
    fig.data[3].visible = True
    # TODO: Show annotations for the first frequency

    # Add slider
    steps = []
    for i, freq in enumerate(data.df["delta_frequency"].unique()):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"annotations": [[False] * len(fig.data), annotations_dict[i]]},
            ],
            label=f"{data.df['delta_frequency'].unique()[i]:.6f}",
        )
        for j in range(4):
            step["args"][0]["visible"][i * 4 + j] = True
            step["args"][1]["annotations"][0][i * 4 + j] = True
        steps.append(step)

    sliders = [
        dict(
            currentvalue={"prefix": "delta_frequency: "},
            steps=steps,
        )
    ]

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="i (V)",
        yaxis_title="q (V)",
        sliders=sliders,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig
