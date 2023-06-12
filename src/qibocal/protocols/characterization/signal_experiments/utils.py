import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Signals #add phase
def signals(data, fit, qubit):
    figures = []
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "State 0",
            "State 1",
        ),
    )

    fitting_report = ""

    states = data.df["state"].unique()
    for state in states:
        state_data = data.df[data.df["state"] == state]
        fig.add_trace(
            go.Scatter(
                x=state_data["sample"],
                y=state_data["MSR"].pint.to("uV").pint.magnitude,
            ),
            row=1,
            col=1 + state,
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Sample",
        yaxis_title="MSR (uV)",
    )

    fitting_report = fitting_report + (f"{qubit} | Time of flight : <br>")

    figures.append(fig)

    return figures, fitting_report


# Integration weights calculation
# add phase ?
def signal_0_1(data, fit, qubit):
    figures = []
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("State 0-1",),
    )

    integration_weights = fit.optimal_integration_weights[qubit]

    fitting_report = ""
    state = "1-0"

    fig.add_trace(
        go.Scatter(
            y=integration_weights,
            name=f"q{qubit}",
            showlegend=not bool(state),
            legendgroup=f"q{qubit}",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            y=np.ones([len(integration_weights)]),
            name=f"q{qubit}",
            showlegend=not bool(state),
            legendgroup=f"q{qubit}",
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Sample",
        yaxis_title="MSR (uV)",
    )

    fitting_report = fitting_report + (f"{qubit} | Optimal integration weights : <br>")

    figures.append(fig)

    return figures, fitting_report
