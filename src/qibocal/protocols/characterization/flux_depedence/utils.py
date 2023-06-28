import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import HZ_TO_GHZ, V_TO_UV


def flux_dependence_plot(data, qubit, fit):
    figures = []
    fitting_report = None
    qubit_data = data[qubit]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR [V]",
            "Phase [rad]",
        ),
    )
    frequencies = qubit_data.freq * HZ_TO_GHZ
    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=qubit_data.bias,
            z=qubit_data.msr * V_TO_UV,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text=f"{qubit}: Frequency (Hz)",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Bias (V)", row=1, col=1)

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=qubit_data.bias,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text=f"{qubit}: Frequency (Hz)",
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Bias (V)", row=1, col=2)

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report
