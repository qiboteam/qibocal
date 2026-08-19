"""Plotting routines for flux dependent protocols."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import HZ_TO_GHZ


def flux_dependence_plot(
    data,
    fit,
    qubit,
    inliers,
    outliers,
    fit_function,
):
    figures = []
    qubit_data = data[qubit]
    frequencies = qubit_data.freq * HZ_TO_GHZ

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=qubit_data.freq * HZ_TO_GHZ,
            y=qubit_data.bias,
            z=qubit_data.signal,
            colorbar={"title": "Signal [a.u.]"},
            colorscale="Viridis",
        ),
    )

    # TODO: This fit is for frequency, can it be reused here, do we even want the fit ?
    if (
        fit is not None
        and fit_function is not None
        and data.__class__.__name__ != "CouplerSpectroscopyData"
        and fit.successful_fit[qubit]
    ):
        params = fit.fitted_parameters[qubit]
        bias = np.unique(qubit_data.bias)
        fig.add_trace(
            go.Scatter(
                x=fit_function(bias, **params) * HZ_TO_GHZ,
                y=bias,
                showlegend=True,
                name="Fit",
                marker={"color": "rgb(248, 248, 248)"},
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    fit.frequency[qubit] * HZ_TO_GHZ,
                ],
                y=[
                    fit.sweetspot[qubit],
                ],
                mode="markers",
                marker={
                    "size": 8,
                    "color": "red",
                },
                name="Sweetspot",
                showlegend=True,
            ),
        )

        # Inliers and outliers plotting for debugging purposes
        if inliers is not None and len(inliers) > 0:
            fig.add_trace(
                go.Scatter(
                    x=inliers[:, 1] * HZ_TO_GHZ,  # frequency
                    y=inliers[:, 0],  # bias
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": "white",
                    },
                    name="Inliers",
                    showlegend=True,
                    visible="legendonly",
                ),
            )

        if outliers is not None and len(outliers) > 0:
            fig.add_trace(
                go.Scatter(
                    x=outliers[:, 1] * HZ_TO_GHZ,  # frequency
                    y=outliers[:, 0],  # bias
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": "green",
                    },
                    name="Outliers",
                    showlegend=True,
                    visible="legendonly",
                ),
            )

    fig.update_xaxes(
        title_text="Frequency [GHz]",
    )
    fig.update_yaxes(title_text="Bias [a.u.]")

    fig.update_layout(xaxis1={"range": [np.min(frequencies), np.max(frequencies)]})

    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h"},
    )

    figures.append(fig)

    return figures


def flux_crosstalk_plot(data, qubit, fit, fit_function):
    figures = []
    fitting_report = ""
    all_qubit_data = {
        index: data_qubit
        for index, data_qubit in data.data.items()
        if index[0] == qubit
    }
    fig = make_subplots(
        rows=1,
        cols=len(all_qubit_data),
        horizontal_spacing=0.3 / len(all_qubit_data),
        vertical_spacing=0.1,
        subplot_titles=len(all_qubit_data) * ("Signal [a.u.]",),
    )
    for col, (flux_qubit, qubit_data) in enumerate(all_qubit_data.items()):
        frequencies = qubit_data.freq * HZ_TO_GHZ
        fig.add_trace(
            go.Heatmap(
                x=frequencies,
                y=qubit_data.bias,
                z=qubit_data.signal,
                showscale=False,
            ),
            row=1,
            col=col + 1,
        )
        if fit is not None and fit.successful_fit[qubit] and flux_qubit[1] != qubit:
            fig.add_trace(
                go.Scatter(
                    x=fit_function(
                        xj=qubit_data.bias, **fit.fitted_parameters[flux_qubit]
                    ),
                    y=qubit_data.bias,
                    showlegend=not any(
                        isinstance(trace, go.Scatter) for trace in fig.data
                    ),
                    legendgroup="Fit",
                    name="Fit",
                    marker={"color": "green"},
                ),
                row=1,
                col=col + 1,
            )

        fig.update_xaxes(
            title_text="Frequency [GHz]",
            row=1,
            col=col + 1,
        )

        fig.update_yaxes(
            title_text=f"Qubit {flux_qubit[1]}: Bias [a.u.]", row=1, col=col + 1
        )

    fig.update_layout(xaxis1={"range": [np.min(frequencies), np.max(frequencies)]})
    fig.update_layout(xaxis2={"range": [np.min(frequencies), np.max(frequencies)]})
    fig.update_layout(xaxis3={"range": [np.min(frequencies), np.max(frequencies)]})
    fig.update_layout(
        showlegend=True,
    )
    figures.append(fig)

    return figures, fitting_report
