from enum import Enum
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

from .....auto.operation import QubitId, QubitPairId
from .....config import log
from ....utils import fallback_period, guess_period
from ..utils import Basis, SetControl
from . import fitting

EPS = 1e-15


class HamiltonianTerm(str, Enum):
    """Hamiltonian terms for CR effective Hamiltonian."""

    IX = "IX"
    IY = "IY"
    IZ = "IZ"
    ZX = "ZX"
    ZY = "ZY"
    ZZ = "ZZ"


def tomography_cr_fit(
    data: Union[
        "HamiltonianTomographyCRLengthData",  # noqa: F821
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
    ],
) -> dict[tuple[QubitId, QubitId, Basis, SetControl], list]:
    """Perform fitting on expectation values for CR tomography.

    We first fit the Z expectation value to get the frequency of the CR pulse.
    We then fit both the X and Y component.
    Finally we perform a simultaneous fit all three components taking into account
    constraint on the parameters.
    """
    fitted_parameters = {}
    for pair in data.pairs:
        for setup in SetControl:
            pair_data = data.data[pair[0], pair[1], Basis.Z, setup]
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
                    maxfev=int(1e6),
                    p0=pguess,
                    sigma=pair_data.error_target,
                    bounds=(
                        [-5 * omega, -5 * omega, -5 * omega, 0],
                        [
                            5 * omega,
                            5 * omega,
                            5 * omega,
                            5 * omega,
                        ],
                    ),
                )
                fitted_parameters[pair[0], pair[1], Basis.Z, setup] = popt.tolist()
            except Exception as e:  # pragma: no cover
                log.warning(f"CR Z fit failed for pair {pair} due to {e}.")

    for pair in data.pairs:
        for setup in SetControl:
            pair_data = data.data[pair[0], pair[1], Basis.X, setup]
            omega = fitted_parameters[pair[0], pair[1], Basis.Z, setup][3]
            pguess = [0, 0, 0, omega]

            try:
                popt, _ = curve_fit(
                    fitting.fit_X_exp,
                    pair_data.x,
                    pair_data.prob_target,
                    maxfev=int(1e6),
                    p0=pguess,
                    sigma=pair_data.error_target,
                    absolute_sigma=True,
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
            except Exception as e:  # pragma: no cover
                log.warning(f"CR fit failed X for pair {pair} due to {e}.")

    for pair in data.pairs:
        for setup in SetControl:
            pair_data = data.data[pair[0], pair[1], Basis.Y, setup]
            omega = fitted_parameters[pair[0], pair[1], Basis.Z, setup][3]
            pguess = [0, 0, 0, omega]
            try:
                popt, _ = curve_fit(
                    fitting.fit_Y_exp,
                    pair_data.x,
                    pair_data.prob_target,
                    maxfev=int(1e6),
                    sigma=pair_data.error_target,
                    absolute_sigma=True,
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
                fitted_parameters[pair[0], pair[1], Basis.Y, setup] = popt.tolist()
            except Exception as e:  # pragma: no cover
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
                        data.data[pair[0], pair[1], Basis.X, setup].prob_target,
                        data.data[pair[0], pair[1], Basis.Y, setup].prob_target,
                        data.data[pair[0], pair[1], Basis.Z, setup].prob_target,
                    ]
                ),
                maxfev=int(1e6),
                p0=pguess,
                sigma=np.concatenate(
                    [
                        data.data[pair[0], pair[1], Basis.X, setup].error_target,
                        data.data[pair[0], pair[1], Basis.Y, setup].error_target,
                        data.data[pair[0], pair[1], Basis.Z, setup].error_target,
                    ]
                ),
            )
            fitted_parameters[pair[0], pair[1], setup] = popt.tolist()
    return fitted_parameters


def extract_hamiltonian_terms(pair: QubitPairId, fitted_parameters: dict) -> dict:
    """Extract Hamiltonian terms from fitted parameters.

    We follow the procedure presented in the paper https://arxiv.org/pdf/2303.01427.
    """
    hamiltonian_terms = {}
    hamiltonian_terms[pair[0], pair[1], HamiltonianTerm.ZX] = 0.5 * (
        fitted_parameters[pair[0], pair[1], SetControl.Id][0]
        - fitted_parameters[pair[0], pair[1], SetControl.X][0]
    )
    hamiltonian_terms[pair[0], pair[1], HamiltonianTerm.IX] = 0.5 * (
        fitted_parameters[pair[0], pair[1], SetControl.Id][0]
        + fitted_parameters[pair[0], pair[1], SetControl.X][0]
    )
    hamiltonian_terms[pair[0], pair[1], HamiltonianTerm.ZY] = 0.5 * (
        fitted_parameters[pair[0], pair[1], SetControl.Id][1]
        - fitted_parameters[pair[0], pair[1], SetControl.X][1]
    )
    hamiltonian_terms[pair[0], pair[1], HamiltonianTerm.IY] = 0.5 * (
        fitted_parameters[pair[0], pair[1], SetControl.Id][1]
        + fitted_parameters[pair[0], pair[1], SetControl.X][1]
    )
    hamiltonian_terms[pair[0], pair[1], HamiltonianTerm.ZZ] = 0.5 * (
        fitted_parameters[pair[0], pair[1], SetControl.Id][2]
        - fitted_parameters[pair[0], pair[1], SetControl.X][2]
    )
    hamiltonian_terms[pair[0], pair[1], HamiltonianTerm.IZ] = 0.5 * (
        fitted_parameters[pair[0], pair[1], SetControl.Id][2]
        + fitted_parameters[pair[0], pair[1], SetControl.X][2]
    )
    return hamiltonian_terms


def compute_total_expectation_value(
    data: Union[
        "HamiltonianTomographyCRLengthData",  # noqa: F821
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
    ],
    pair: QubitPairId,
):
    tot_exp_vals = []
    for basis in Basis:
        tot_exp_vals.append(
            data.data[pair[0], pair[1], basis, SetControl.Id].prob_target
            + data.data[pair[0], pair[1], basis, SetControl.X].prob_target
        )
    return np.vstack(tot_exp_vals)


def bloch_func(x, pair: QubitPairId, fitted_parameters: dict):
    x = np.vstack([x, x, x])
    id_blochfit = fitting.combined_fit(
        x, *fitted_parameters[pair[0], pair[1], SetControl.Id]
    ).reshape((3, -1))
    x_blochfit = fitting.combined_fit(
        x, *fitted_parameters[pair[0], pair[1], SetControl.X]
    ).reshape((3, -1))
    return np.sqrt(np.sum((id_blochfit + x_blochfit) ** 2, axis=0))


def compute_bloch_vector(
    data: Union[
        "HamiltonianTomographyCRLengthData",  # noqa: F821
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
    ],
    pair: QubitPairId,
    fitted_parameters: dict = None,
):
    bloch_exp = compute_total_expectation_value(data, pair)
    bloch_exp = np.sqrt(np.sum((bloch_exp) ** 2, axis=0))

    bloch_fit = None
    if fitted_parameters is not None:
        times = data.data[pair[0], pair[1], Basis.Z, SetControl.Id].x
        times_range = np.linspace(min(times), max(times), 2 * len(times))

        bloch_fit = bloch_func(times_range, pair, fitted_parameters)

    return bloch_exp, bloch_fit


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
) -> tuple[list[go.Figure], str]:
    """Plotting function for HamiltonianTomographyCRLength."""
    fig = make_subplots(
        rows=4,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.05,
        shared_xaxes=True,
        shared_yaxes=True,
    )
    for i, basis in enumerate(Basis):
        for setup in SetControl:
            target = target if target in data.pairs else (target[1], target[0])
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
                            name=f"Single target when control at {0 if setup is SetControl.Id else 1}",
                            showlegend=True if basis is Basis.Z else False,
                            legendgroup=f"Single target when control at {0 if setup is SetControl.Id else 1}",
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
                            name=f"Single target when control at {0 if setup is SetControl.Id else 1}",
                            showlegend=True if basis is Basis.Z else False,
                            legendgroup=f"Single target when control at {0 if setup is SetControl.Id else 1}",
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
                            name=f"Single target when control at {0 if setup is SetControl.Id else 1}",
                            showlegend=True if basis is Basis.Z else False,
                            legendgroup=f"Single target when control at {0 if setup is SetControl.Id else 1}",
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
                        y=getattr(fitting, f"fit_{basis.name}_exp")(
                            x,
                            wx=fit.fitted_parameters[target[0], target[1], setup][0],
                            wy=fit.fitted_parameters[target[0], target[1], setup][1],
                            wz=fit.fitted_parameters[target[0], target[1], setup][2],
                            w=np.sqrt(
                                np.sum(
                                    i**2
                                    for i in fit.fitted_parameters[
                                        target[0], target[1], setup
                                    ]
                                )
                            ),
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

    bloch_vect, bloch_fit = compute_bloch_vector(data, target, fit.fitted_parameters)
    fig.add_trace(
        go.Scatter(
            x=pair_data.x,
            y=bloch_vect,
            name="Bloch vector |R(t)|",
            legendgroup="Bloch vector |R(t)|",
            showlegend=True,
            mode="markers",
        ),
        row=4,
        col=1,
    )
    if bloch_fit is not None:
        x = np.linspace(pair_data.x.min(), pair_data.x.max(), len(bloch_fit))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bloch_fit,
                name="Fitted Bloch vector |R(t)|",
                showlegend=True,
                legendgroup="Fitted Bloch vector |R(t)|",
                mode="lines",
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        yaxis1=dict(range=[-1.2, 1.2]),
        yaxis2=dict(range=[-1.2, 1.2]),
        yaxis3=dict(range=[-1.2, 1.2]),
        yaxis4=dict(range=[-0.2, 2.2]),
        height=800,
    )
    fig.update_yaxes(title_text="<X(t)>", row=1, col=1)
    fig.update_yaxes(title_text="<Y(t)>", row=2, col=1)
    fig.update_yaxes(title_text="<Z(t)>", row=3, col=1)
    fig.update_yaxes(title_text="|R(t)|", row=4, col=1)

    return [fig], ""
