from enum import Enum
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import (
    Bounds,
    NonlinearConstraint,
    curve_fit,
    differential_evolution,
)

from .....auto.operation import (
    Data,
    QubitId,
    QubitPairId,
)
from .....config import log
from ....utils import quinn_fernandes_algorithm, table_dict, table_html
from ..utils import Basis, SetControl
from . import fitting

EPS = 1e-15
QUANTILE_CONSTANT = 1.5


class HamiltonianTerm(str, Enum):
    """Hamiltonian terms for CR effective Hamiltonian."""

    IX = "IX"
    IY = "IY"
    IZ = "IZ"
    ZX = "ZX"
    ZY = "ZY"
    ZZ = "ZZ"


def cyclic_prob(values: float, state: int):
    return np.minimum(values, 1) if state == 1 else np.maximum(1 - values, 0)


def dynamic_evolution_optimizer(
    signals_id: np.ndarray,
    x: np.ndarray,
    concat_x: np.ndarray,
    init_omega_guess: float,
    use_constraints: bool = False,
) -> np.ndarray:
    """Optimizer for sinusoidal fitting; it exploits evolution algorithm to find the best fit parameters.
    This algorithm is a gradient-free optimization, hence might be less affected by local minima in the cost
    function landscape, but it can be computationally expensive, especially for large datasets or complex models.
    """

    assert x.ndim == 1, f"Expected 1D array, got array with shape {x.shape}"

    def func_to_minimize(z):
        return np.sum((fitting.combined_fit(concat_x, *z) - signals_id.ravel()) ** 2)

    def constraint(x):
        return np.sqrt(np.sum(x**2))

    bounds = Bounds([0, 0, 0], [init_omega_guess, init_omega_guess, init_omega_guess])
    res = differential_evolution(
        func_to_minimize,
        bounds,
        maxiter=int(1e6),
        constraints=(
            NonlinearConstraint(constraint, 0, init_omega_guess * 2)
            if use_constraints
            else ()
        ),
    )
    popt = res.x
    return popt


def scipy_curve_fit_optimizer(
    concatenated_signal, vector_x, init_omega_guess, errors=None
):
    offsets = abs(np.median(concatenated_signal, axis=-1)) * init_omega_guess**2
    omega_guesses = np.zeros(offsets.shape)
    omega_guesses[-1] = np.sqrt(offsets[-1])
    omega_guesses[0] = offsets[0] / omega_guesses[-1]
    omega_guesses[1] = offsets[1] / omega_guesses[-1]

    popt, _ = curve_fit(
        fitting.combined_fit,
        np.concatenate([vector_x, vector_x, vector_x]),
        concatenated_signal.ravel(),
        maxfev=int(1e6),
        p0=np.maximum(omega_guesses, 0),
        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        sigma=errors,
    )

    return popt


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
            concatenated_signal = np.concatenate(
                [
                    data.data[pair[0], pair[1], Basis.X, setup].prob_target,
                    data.data[pair[0], pair[1], Basis.Y, setup].prob_target,
                    data.data[pair[0], pair[1], Basis.Z, setup].prob_target,
                ]
            ).reshape(len(Basis), -1)

            # concatenated_errors = np.concatenate(
            #     [
            #         data.data[pair[0], pair[1], Basis.X, setup].error_target,
            #         data.data[pair[0], pair[1], Basis.Y, setup].error_target,
            #         data.data[pair[0], pair[1], Basis.Z, setup].error_target,
            #     ]
            # )

            vector_x = data.data[pair[0], pair[1], Basis.X, setup].x

            sampling_rate = 1 / abs(vector_x[1] - vector_x[0])
            total_omega_guess = quinn_fernandes_algorithm(
                concatenated_signal, vector_x, sampling_rate, speedup_flag=True
            )

            # popt = scipy_curve_fit_optimizer(concatenated_signal, vector_x, total_omega_guess, concatenated_errors)
            popt = dynamic_evolution_optimizer(
                concatenated_signal,
                vector_x,
                np.concatenate([vector_x, vector_x, vector_x]),
                total_omega_guess,
            )

            fitted_parameters[pair[0], pair[1], setup] = popt.tolist()
    return fitted_parameters


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


def refactor_hamiltoanian_terms(ham_terms: dict, pair: QubitPairId):
    ham_terms[HamiltonianTerm.IX] = ham_terms.pop(
        (pair[0], pair[1], HamiltonianTerm.IX)
    )
    ham_terms[HamiltonianTerm.IY] = ham_terms.pop(
        (pair[0], pair[1], HamiltonianTerm.IY)
    )
    ham_terms[HamiltonianTerm.IZ] = ham_terms.pop(
        (pair[0], pair[1], HamiltonianTerm.IZ)
    )
    ham_terms[HamiltonianTerm.ZX] = ham_terms.pop(
        (pair[0], pair[1], HamiltonianTerm.ZX)
    )
    ham_terms[HamiltonianTerm.ZY] = ham_terms.pop(
        (pair[0], pair[1], HamiltonianTerm.ZY)
    )
    ham_terms[HamiltonianTerm.ZZ] = ham_terms.pop(
        (pair[0], pair[1], HamiltonianTerm.ZZ)
    )

    return ham_terms


def amp_tom_fit(x, y, q_pair, term, result_dict):
    try:
        pguess = [0, 0]
        popt, _ = curve_fit(
            fitting.linear_fit,
            x,
            y,
            p0=pguess,
            maxfev=int(1e6),
            absolute_sigma=True,
            bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
        )
        term_fit = popt.tolist()
        result_dict[term] = term_fit
    except Exception as e:
        log.warning(f"{term} term vs amplitudes fit failed for {q_pair} due to {e}.")

    return result_dict


def amplitude_tomography_cr_fit(data: Data):
    amp_hamiltonian_params = {}
    for amp in data.amplitudes:
        amp_data = data.select_amplitude(amp)
        amp_length_params = tomography_cr_fit(amp_data)
        for pair in amp_data.pairs:
            terms = extract_hamiltonian_terms(pair, amp_length_params)
            terms = refactor_hamiltoanian_terms(terms, pair)
            res_tuple = (amp, terms)
            if pair not in amp_hamiltonian_params:
                amp_hamiltonian_params[pair] = [res_tuple]
            else:
                amp_hamiltonian_params[pair].append(res_tuple)

    amp_lin_fit_params = {}
    for pair_key, pair_value in amp_hamiltonian_params.items():
        num_terms = [[] for _ in range(6)]
        amplitudes = []
        for vals in pair_value:
            amplitudes.append(vals[0])
            for n, t in zip(num_terms, HamiltonianTerm):
                n.append(vals[1][t])

        fit_params_pair = {}
        for nterm, t in zip(num_terms, HamiltonianTerm):
            fit_params_pair = amp_tom_fit(
                x=amplitudes,
                y=nterm,
                q_pair=pair_key,
                term=t,
                result_dict=fit_params_pair,
            )

        amp_lin_fit_params[pair_key] = fit_params_pair

    return amp_length_params, amp_hamiltonian_params, amp_lin_fit_params


def phase_tom_fit(x, y, q_pair, term, result_dict):
    fs = 1 / (x[1] - x[0])
    omega = quinn_fernandes_algorithm(y, x, fs)
    median_sig = np.median(y)
    q80 = np.quantile(y, 0.8)
    q20 = np.quantile(y, 0.2)
    amplitude_guess = abs(q80 - q20) / QUANTILE_CONSTANT
    phase_guess = 0
    pguess = [amplitude_guess, median_sig, omega, phase_guess]
    try:
        popt, _ = curve_fit(
            fitting.sin_fit,
            x,
            y,
            p0=pguess,
            maxfev=int(1e6),
            absolute_sigma=True,
            bounds=(
                [-np.inf, -np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf, np.inf],
            ),
        )
        term_fit = popt.tolist()
        result_dict[term] = term_fit
    except Exception as e:
        log.warning(f"{term} term vs amplitudes fit failed for {q_pair} due to {e}.")

    return result_dict


def phase_tomography_cr_fit(data: Data):
    phase_hamiltonian_params = {}
    for phase in data.phases:
        phase_data = data.select_phase(phase)
        phase_length_params = tomography_cr_fit(phase_data)
        for pair in phase_data.pairs:
            terms = extract_hamiltonian_terms(pair, phase_length_params)
            terms = refactor_hamiltoanian_terms(terms, pair)
            res_tuple = (phase, terms)
            if pair not in phase_hamiltonian_params:
                phase_hamiltonian_params[pair] = [res_tuple]
            else:
                phase_hamiltonian_params[pair].append(res_tuple)

    phase_sin_fit_params = {}
    for pair_key, pair_value in phase_hamiltonian_params.items():
        num_terms = [[] for _ in range(6)]
        phases = []
        for vals in pair_value:
            phases.append(vals[0])
            for n, t in zip(num_terms, HamiltonianTerm):
                n.append(vals[1][t])

        fit_params_pair = {}
        for nterm, t in zip(num_terms, HamiltonianTerm):
            fit_params_pair = phase_tom_fit(
                x=phases, y=nterm, q_pair=pair_key, term=t, result_dict=fit_params_pair
            )

        phase_sin_fit_params[pair_key] = fit_params_pair

    return phase_length_params, phase_hamiltonian_params, phase_sin_fit_params


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
    target = target if target in data.pairs else (target[1], target[0])
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
            if fit is not None and (*target, setup) in fit.fitted_parameters:
                x = np.linspace(pair_data.x.min(), pair_data.x.max(), 100)
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


def calibration_cr_plot(
    data: Union[
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
        "HamiltonianTomographyCRPhaseData",  # noqa: F821
    ],
    target: QubitPairId,
    fit: Optional[
        Union[
            "HamiltonianTomographyCRAmplitudeResults",  # noqa: F821
            "HamiltonianTomographyCRPhaseResults",  # noqa: F821
        ]
    ] = None,
) -> tuple[list[go.Figure], str]:
    """Plotting function for HamiltonianTomographyCRLength."""
    figures = []
    fitting_report = ""

    if type(data).__name__ == "HamiltonianTomographyCRAmplitudeData":
        fit_func = fitting.linear_fit
        x_title = "amplitude [a.u.]"
        for a in data.amplitude_plot_dict[target]:
            if a in data.amplitudes:
                amp_data = data.select_amplitude(a)
                amp_fig, _ = tomography_cr_plot(
                    amp_data, target, fit.tomography_length_parameters
                )
                figures.append(amp_fig[0])
            else:
                log.warning("inserted non existing amplitudes values to plot.")

    if type(data).__name__ == "HamiltonianTomographyCRPhaseData":
        fit_func = fitting.sin_fit
        x_title = "phase [rad.]"
        for p in data.phase_plot_dict[target]:
            if p in data.phases:
                phase_data = data.select_phase(p)
                phase_fig, _ = tomography_cr_plot(
                    phase_data, target, fit.tomography_length_parameters
                )
                figures.append(phase_fig[0])
            else:
                log.warning("inserted non existing amplitudes values to plot.")

    if fit is not None:
        for t in HamiltonianTerm:
            eff_ham_term = [f[1][t] for f in fit.hamiltonian_terms[target]]
            exp_sweeper = [a[0] for a in fit.hamiltonian_terms[target]]
            fig = go.Figure(
                [
                    go.Scatter(
                        x=exp_sweeper,
                        y=eff_ham_term,
                        opacity=1,
                        name=f"{t.name}",
                        showlegend=True,
                        legendgroup="Probability",
                        mode="lines",
                    )
                ]
            )

            if (
                not fit.fitted_parameters
                and target in fit.fitted_parameters
                and t in fit.fitted_parameters[target]
            ):
                amp_range = np.linspace(
                    min(exp_sweeper),
                    max(exp_sweeper),
                    2 * len(exp_sweeper),
                )
                params = fit.fitted_parameters[target][t]
                fig.add_trace(
                    go.Scatter(
                        x=amp_range,
                        y=fit_func(amp_range, *params),
                        name=f"{t.name} Fit",
                        line=go.scatter.Line(dash="dot"),
                        marker_color="rgb(255, 130, 67)",
                    ),
                )

                fig.update_layout(
                    showlegend=True,
                    xaxis_title=(
                        f"Target cancellation {x_title}"
                        if fit.cancellation_calibration
                        else f"Control drive {x_title}"
                    ),
                    yaxis_title="Interaction strength [MHz]",
                )

                fitting_report = table_html(
                    table_dict(
                        6 * [target],
                        [f"{term.name}: Fitted_parameters" for term in HamiltonianTerm],
                        [
                            fit.fitted_parameters[target][term]
                            for term in HamiltonianTerm
                        ],
                    )
                )

            figures.append(fig)

    return figures, fitting_report
