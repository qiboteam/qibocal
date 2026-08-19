import numpy as np
import plotly.graph_objects as go

from qibocal.auto.operation import QubitId

from ..utils import COLORBAND, COLORBAND_LINE, table_dict, table_html
from .fitting import exp_decay


def plot(data, target: QubitId, fit=None) -> tuple[list[go.Figure], str]:
    """Plotting function for spin-echo or CPMG protocol."""

    figures = []
    fitting_report = ""
    qubit_data = data[target]
    waits = qubit_data.wait
    probs = qubit_data.prob
    error_bars = qubit_data.error

    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=probs,
                opacity=1,
                name="Probability of 1",
                showlegend=True,
                legendgroup="Probability of 1",
                mode="markers",
            ),
            go.Scatter(
                x=np.concatenate((waits, waits[::-1])),
                y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line={"color": COLORBAND_LINE},
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    if fit is not None:
        waitrange = np.linspace(
            min(waits),
            max(waits),
            2 * len(qubit_data),
        )
        params = fit.fitted_parameters[target]

        fig.add_trace(
            go.Scatter(
                x=waitrange,
                y=exp_decay(waitrange, *params),
                name="Fit",
                mode="lines",
            ),
        )
        fitting_report = table_html(
            table_dict(
                target,
                ["T2", "chi2 reduced"],
                [fit.t2[target], fit.chi2[target]],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Probability of State 1",
    )

    figures.append(fig)

    return figures, fitting_report
