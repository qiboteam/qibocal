from typing import Callable, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

from .....auto.operation import QubitId, QubitPairId
from .....config import log
from ....utils import fallback_period, guess_period
from ..utils import Basis, SetControl
from . import fitting


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
            pair_data = data[pair[0], pair[1], Basis.Z, setup]
            period = fallback_period(guess_period(pair_data.x, pair_data.prob_target))
            omega = 2 * np.pi / period
            pguess = [
                omega / np.sqrt(2),
                omega / np.sqrt(2),
                0,
                omega,
            ]
            try:
                popt, _ = curve_fit(
                    fitting.fit_Z_exp,
                    pair_data.x,
                    pair_data.prob_target,
                    maxfev=10000,
                    p0=pguess,
                    sigma=pair_data.error_target,
                    bounds=(
                        [0, 0, 0, 0],
                        [
                            5 * omega,
                            5 * omega,
                            1,
                            5 * omega,
                        ],
                    ),
                )
                fitted_parameters[pair[0], pair[1], Basis.Z, setup] = popt.tolist()
            except Exception as e:
                log.warning(f"CR Z fit failed for pair {pair} due to {e}.")

    for pair in data.pairs:
        for setup in SetControl:
            pair_data = data[pair[0], pair[1], Basis.X, setup]
            pguess = fitted_parameters[pair[0], pair[1], Basis.Z, setup]
            omega = fitted_parameters[pair[0], pair[1], Basis.Z, setup][3]
            try:
                popt, _ = curve_fit(
                    fitting.fit_X_exp,
                    pair_data.x,
                    pair_data.prob_target,
                    maxfev=10000,
                    p0=pguess,
                    sigma=pair_data.error_target,
                    bounds=(
                        [-omega, -omega, -omega, 0.99 * omega],
                        [
                            omega,
                            omega,
                            omega,
                            1.01 * omega,
                        ],
                    ),
                )
                fitted_parameters[pair[0], pair[1], Basis.X, setup] = popt.tolist()
            except Exception as e:
                log.warning(f"CR fit failed X for pair {pair} due to {e}.")

    for pair in data.pairs:
        for setup in SetControl:
            pair_data = data[pair[0], pair[1], Basis.Y, setup]
            pguess = fitted_parameters[pair[0], pair[1], Basis.X, setup]
            omega = fitted_parameters[pair[0], pair[1], Basis.Z, setup][3]
            try:
                popt, _ = curve_fit(
                    fitting.fit_Y_exp,
                    pair_data.x,
                    pair_data.prob_target,
                    maxfev=10000,
                    sigma=pair_data.error_target,
                    bounds=(
                        [
                            -omega if setup is SetControl.Id else -0.1 * omega,
                            1.1 * pguess[1] if pguess[1] < 0 else 0.9 * pguess[1],
                            -omega if setup is SetControl.X else -0.1 * omega,
                            0.99 * omega,
                        ],
                        [
                            0.1 * omega if setup is SetControl.Id else omega,
                            1.1 * pguess[1] if pguess[1] > 0 else 0.9 * pguess[1],
                            0.1 * omega if setup is SetControl.X else omega,
                            1.01 * omega,
                        ],
                    ),
                )
                fitted_parameters[pair[0], pair[1], Basis.Y, setup] = popt.tolist()
            except Exception as e:
                log.warning(f"CR Y fit failed for pair {pair} due to {e}.")

    for pair in data.pairs:
        for setup in SetControl:
            fitted_parameters[pair[0], pair[1], setup] = fitted_parameters[
                pair[0], pair[1], Basis.Y, setup
            ][:3]
            pguess = fitted_parameters[pair[0], pair[1], setup]
            popt, _ = curve_fit(
                fitting.combined_fit,
                np.concatenate([pair_data.x, pair_data.x, pair_data.x]),
                np.concatenate(
                    [
                        data[pair[0], pair[1], Basis.X, setup].prob_target,
                        data[pair[0], pair[1], Basis.Y, setup].prob_target,
                        data[pair[0], pair[1], Basis.Z, setup].prob_target,
                    ]
                ),
                maxfev=10000,
                p0=pguess,
                sigma=np.concatenate(
                    [
                        data[pair[0], pair[1], Basis.X, setup].error_target,
                        data[pair[0], pair[1], Basis.Y, setup].error_target,
                        data[pair[0], pair[1], Basis.Z, setup].error_target,
                    ]
                ),
            )
            fitted_parameters[pair[0], pair[1], setup] = popt.tolist()
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
                if basis == Basis.Z:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=fitting.fit_Z_exp(
                                x,
                                *fit.fitted_parameters[
                                    target[0], target[1], basis, setup
                                ],
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
                elif basis == Basis.X:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=fitting.fit_X_exp(
                                x,
                                *fit.fitted_parameters[
                                    target[0], target[1], basis, setup
                                ],
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
                elif basis == Basis.Y:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=fitting.fit_Y_exp(
                                x,
                                *fit.fitted_parameters[
                                    target[0], target[1], basis, setup
                                ],
                            ),
                            name=f"Single fit of target when control at {0 if setup is SetControl.Id else 1}",
                            showlegend=True if basis is Basis.Z else False,
                            legendgroup=f"Single Fit target when control at {0 if setup is SetControl.Id else 1}",
                            mode="lines",
                            line=dict(
                                color="blue" if setup is SetControl.Id else "red",
                            ),
                        ),
                        row=i + 1,
                        col=1,
                    )

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=getattr(fitting, f"fit_{basis.name}_exp_fine")(
                            x,
                            *fit.fitted_parameters[target[0], target[1], setup],
                        ),
                        name=f"Simultaneous Fit of target when control at {0 if setup is SetControl.Id else 1}",
                        showlegend=True if basis is Basis.Z else False,
                        legendgroup=f"Simultaneous Fit target when control at {0 if setup is SetControl.Id else 1}",
                        mode="lines",
                        line=dict(
                            color="green" if setup is SetControl.Id else "orange",
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
