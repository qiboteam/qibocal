from typing import Callable, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .....auto.operation import QubitId, QubitPairId
from .....config import log
from ....utils import fallback_period, guess_period
from ..utils import Basis, SetControl


def tomography_cr_fit(
    data: Union[
        "HamiltonianTomographyCRLengthData",  # noqa: F821
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
    ],
    fitting_function: Callable,
) -> dict[tuple[QubitId, QubitId, Basis, SetControl], list]:
    fitted_parameters = {}
    for pair in data.pairs:
        for setup in SetControl:
            for basis in Basis:
                pair_data = data[pair[0], pair[1], basis, setup]
                raw_x = pair_data.x
                min_x = np.min(raw_x)
                max_x = np.max(raw_x)
                y = pair_data.prob_target
                x = (raw_x - min_x) / (max_x - min_x)

                period = fallback_period(guess_period(x, y))
                pguess = (
                    [0, 0.5, period, 0, 0]
                    if fitting_function.__name__ == "fit_length_function"
                    else [0, 0.5, period, 0]
                )

                try:
                    popt, _, _ = fitting_function(
                        x,
                        y,
                        pguess,
                        sigma=pair_data.error_target,
                        signal=False,
                        x_limits=(min_x, max_x),
                    )
                    fitted_parameters[pair[0], pair[1], basis, setup] = popt
                except Exception as e:
                    log.warning(f"CR fit failed for pair {pair} due to {e}.")
    return fitted_parameters


def tomography_cr_plot(
    data: Union[
        "HamiltonianTomographyCRLengthData",  # noqa: F821
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
    ],
    target: QubitPairId,
    fit: Optional[
        Union[
            "HamiltonianTomographyCRLengthResults",  # noqa: F821
            "HamiltonianTomographyCRAmplitudeResults",  # noqa: F821
        ]
    ] = None,
    fitting_function: Optional[Callable] = None,
) -> tuple[list[go.Figure], str]:
    fig = make_subplots(
        rows=3,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.05,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for i, basis in enumerate(Basis):
        for setup in SetControl:
            pair_data = data.data[target[0], target[1], basis, setup]
            fig.add_trace(
                go.Scatter(
                    x=pair_data.x,
                    y=pair_data.prob_target,
                    name=f"Target when Control at {0 if setup is SetControl.Id else 1}",
                    showlegend=True if basis is Basis.Z else False,
                    legendgroup=f"Target when Control at {0 if setup is SetControl.Id else 1}",
                    mode="markers",
                    marker=dict(color="blue" if setup is SetControl.Id else "red"),
                    error_y=dict(
                        type="data",
                        array=pair_data.error_target,
                        visible=True,
                    ),
                ),
                row=i + 1,
                col=1,
            )
            if fit is not None:
                x = np.linspace(pair_data.x.min(), pair_data.x.max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=fitting_function(
                            x,
                            *fit.fitted_parameters[target[0], target[1], basis, setup],
                        ),
                        name=f"Fit target when control at {0 if setup is SetControl.Id else 1}",
                        showlegend=True if basis is Basis.Z else False,
                        legendgroup=f"Fit target when control at {0 if setup is SetControl.Id else 1}",
                        mode="lines",
                        line=dict(
                            color="blue" if setup is SetControl.Id else "red",
                        ),
                    ),
                    row=i + 1,
                    col=1,
                )

    fig.update_layout(
        yaxis1=dict(range=[-1.2, 1.2]),
        yaxis2=dict(range=[-1.2, 1.2]),
        yaxis3=dict(range=[-1.2, 1.2]),
        height=600,
    )
    fig.update_yaxes(title_text="<X(t)>", row=1, col=1)
    fig.update_yaxes(title_text="<Y(t)>", row=2, col=1)
    fig.update_yaxes(title_text="<Z(t)>", row=3, col=1)

    return [fig], ""
