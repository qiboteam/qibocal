from enum import Enum
from typing import Any, Callable, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import (
    Bounds,
    NonlinearConstraint,
    curve_fit,
    differential_evolution,
    minimize,
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

    bounds = Bounds([-init_omega_guess] * 3, [init_omega_guess] * 3)
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
    sampling_rate = 1 / abs(vector_x[1] - vector_x[0])
    period = int(2 * np.pi / init_omega_guess * sampling_rate)

    offsets = np.median(concatenated_signal[:, :period], axis=-1) * init_omega_guess**2

    omega_guesses = np.zeros(offsets.shape)
    omega_guesses[-1] = np.sqrt(np.abs(offsets[-1]))
    omega_guesses[1] = offsets[1] / omega_guesses[-1]
    omega_guesses[0] = offsets[0] / omega_guesses[-1]
    omega_guesses = np.concatenate([omega_guesses, [0]])

    popt, _ = curve_fit(
        fitting.combined_fit,
        np.concatenate([vector_x, vector_x, vector_x]),
        concatenated_signal.ravel(),
        maxfev=int(1e6),
        p0=omega_guesses,
        bounds=([-np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]),
        sigma=errors,
    )

    return popt


def numerical_root_finder(
    root_func,
    x_range: Union[np.ndarray, list],
    tol: float,
    **kwargs,
):
    def func_to_solve(x):
        return np.abs(root_func(x, **kwargs))

    x_grid = np.linspace(x_range[0], x_range[-1], 100 * len(x_range))

    y_vals = np.array([func_to_solve(xi) for xi in x_grid]).flatten()

    cr_sig = x_grid[y_vals - np.min(y_vals) <= tol][0]

    # Use a numerical minimizer starting from our guess
    res = minimize(
        func_to_solve,
        x0=cr_sig,
        method="Nelder-Mead",
    )

    if res.success:
        cr_sig = min(res.x)

    return cr_sig


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
    fitted_parameters: dict | None = None,
):
    bloch_exp = compute_total_expectation_value(data, pair)
    bloch_exp = np.sqrt(np.sum((bloch_exp) ** 2, axis=0))

    bloch_fit = None
    if fitted_parameters is not None:
        times = data.data[pair[0], pair[1], Basis.Z, SetControl.Id].x
        times_range = np.linspace(min(times), max(times), 2 * len(times))

        bloch_fit = bloch_func(times_range, pair, fitted_parameters)

    return bloch_exp, bloch_fit


def estimate_cr_param(
    x_range,
    data: Union[
        "HamiltonianTomographyCRLengthData",  # noqa: F821
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
    ],
    pair: QubitPairId,
    fitted_parameters: dict,
    tol: float = 1e-6,
):
    bloch_data, _ = compute_bloch_vector(data, pair)
    idx = np.argmin(bloch_data)
    param = x_range[idx]

    if all([(pair[0], pair[1], s) in fitted_parameters for s in SetControl]):
        bloch_data, bloch_fit = compute_bloch_vector(data, pair, fitted_parameters)
        x_range = data.data[pair[0], pair[1], Basis.Z, SetControl.Id].x
        if min(bloch_data) >= min(bloch_fit):
            param = numerical_root_finder(
                root_func=bloch_func,
                x_range=x_range,
                tol=tol,
                pair=pair,
                fitted_parameters=fitted_parameters,
            )

    if type(data).__name__ == "HamiltonianTomographyCRLengthData":
        return int(param)

    return float(param)


def tune_cancellation_sequence(
    x,
    function_to_tune: Callable[Any, float],
    interactions_to_analyze: list[HamiltonianTerm],
    ham_term: dict,
    fit_params: dict,
    tuned_keys: list[str],
    tol: float,
):
    assert len(tuned_keys) == len(interactions_to_analyze), (
        """tuned_keys and interactions_to_analyze must be equally long."""
    )

    # converting list into numpy array
    x = np.array(x)

    tuned_parameters = {}
    for ham_int, k in zip(interactions_to_analyze, tuned_keys):
        selected_ham_term = np.abs(np.array(ham_term[ham_int]))
        min_idx = np.argmin(selected_ham_term)
        tuned_parameters[k] = float(x[min_idx])
        if ham_int in fit_params:
            x_fit = function_to_tune(x, **fit_params[ham_int])
            if np.min(np.abs(selected_ham_term)) >= np.min(np.abs(x_fit)):
                tuned_parameters[k] = float(
                    numerical_root_finder(
                        root_func=function_to_tune,
                        x_range=x,
                        tol=tol,
                        **fit_params[ham_int],
                    )
                )

    return tuned_parameters


def estimate_cancellation_amplitudes(
    amplitudes,
    ham_term: dict,
    ampl_params: dict,
    tol: float = 1e-6,
):
    interaction_terms = [HamiltonianTerm.IX, HamiltonianTerm.IY]
    phases_names = ["ampl_ix", "ampl_iy"]

    tuned_amplitudes = tune_cancellation_sequence(
        x=amplitudes,
        function_to_tune=fitting.linear_fit,
        interactions_to_analyze=interaction_terms,
        ham_term=ham_term,
        fit_params=ampl_params,
        tuned_keys=phases_names,
        tol=tol,
    )

    return tuned_amplitudes


def estimate_cr_phases(
    phases,
    ham_term: dict,
    phase_params: dict,
    tol: float = 1e-6,
):
    interaction_terms = [HamiltonianTerm.ZY, HamiltonianTerm.IY]
    phases_names = ["phi0", "phi1"]

    tuned_phases = tune_cancellation_sequence(
        x=phases,
        function_to_tune=fitting.sin_fit,
        interactions_to_analyze=interaction_terms,
        ham_term=ham_term,
        fit_params=phase_params,
        tuned_keys=phases_names,
        tol=tol,
    )

    if fitting.sin_fit(tuned_phases["phi0"], **phase_params[HamiltonianTerm.ZX]) > 0:
        # in https://journals.aps.org/pra/pdf/10.1103/PhysRevA.93.060302
        # it is said we need to choose the CR phase that minimizes ZY components
        # and maximizes ZX interaction; though we compensate it with the final
        # X rotation on target qubit for building CNOT (otherwise add a n.pi
        # relative phase on X gate)
        tuned_phases["phi0"] += np.pi

    if fitting.sin_fit(tuned_phases["phi1"], **phase_params[HamiltonianTerm.IX]) > 0:
        # same as above, but this time is more euristic.
        tuned_phases["phi1"] += np.pi

    return tuned_phases["phi0"], tuned_phases["phi0"] - tuned_phases["phi1"]


def tomography_cr_fit(
    data: Union[
        "HamiltonianTomographyCRLengthData",  # noqa: F821
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
    ],
    fit_with_evolution: bool = False,
) -> tuple[
    dict[tuple[QubitId, QubitId, SetControl], list], dict[tuple[QubitId, QubitId], int]
]:
    """Perform fitting on expectation values for CR tomography.

    We first fit the Z expectation value to get the frequency of the CR pulse.
    We then fit both the X and Y component.
    Finally we perform a simultaneous fit all three components taking into account
    constraint on the parameters.
    """

    fitted_parameters = {}
    cr_gate_x = {}

    for pair in data.pairs:
        vector_x = data.data[pair[0], pair[1], Basis.X, SetControl.Id].x
        sampling_rate = 1 / abs(vector_x[1] - vector_x[0])
        for setup in SetControl:
            concatenated_signal = np.concatenate(
                [
                    data.data[pair[0], pair[1], Basis.X, setup].prob_target,
                    data.data[pair[0], pair[1], Basis.Y, setup].prob_target,
                    data.data[pair[0], pair[1], Basis.Z, setup].prob_target,
                ]
            ).reshape(len(Basis), -1)

            concatenated_errors = np.concatenate(
                [
                    data.data[pair[0], pair[1], Basis.X, setup].error_target,
                    data.data[pair[0], pair[1], Basis.Y, setup].error_target,
                    data.data[pair[0], pair[1], Basis.Z, setup].error_target,
                ]
            )

            total_omega_guess = quinn_fernandes_algorithm(
                concatenated_signal, vector_x, sampling_rate, speedup_flag=True
            )

            if fit_with_evolution:
                popt = dynamic_evolution_optimizer(
                    concatenated_signal,
                    vector_x,
                    np.concatenate([vector_x, vector_x, vector_x]),
                    total_omega_guess,
                )
            else:
                popt = scipy_curve_fit_optimizer(
                    concatenated_signal,
                    vector_x,
                    total_omega_guess,
                    concatenated_errors,
                )

            fitted_parameters[pair[0], pair[1], setup] = popt.tolist()

        cr_gate_x[pair[0], pair[1]] = estimate_cr_param(
            vector_x, data, pair, fitted_parameters
        )

    return fitted_parameters, cr_gate_x


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
        result_dict[term] = {"a": popt[0], "b": popt[1]}

    except Exception as e:
        log.warning(f"{term} term vs amplitudes fit failed for {q_pair} due to {e}.")

    return result_dict


def amplitude_tomography_cr_fit(data: Data):
    amp_hamiltonian_params = {}
    for amp in data.amplitudes:
        amp_data = data.select_amplitude(amp)
        amp_length_params, cr_duration = tomography_cr_fit(amp_data)
        for pair in amp_data.pairs:
            terms = extract_hamiltonian_terms(pair, amp_length_params)
            terms = refactor_hamiltoanian_terms(terms, pair)
            res_tuple = (amp, terms)
            if pair not in amp_hamiltonian_params:
                amp_hamiltonian_params[pair] = [res_tuple]
            else:
                amp_hamiltonian_params[pair].append(res_tuple)

    amp_lin_fit_params = {}
    cancellating_amplitudes = {}
    for pair_key, pair_value in amp_hamiltonian_params.items():
        num_terms = num_terms = {t: [] for t in HamiltonianTerm}
        amplitudes = []
        for vals in pair_value:
            amplitudes.append(vals[0])
            for t in HamiltonianTerm:
                num_terms[t].append(vals[1][t])

        fit_params_pair = {}
        for t in HamiltonianTerm:
            fit_params_pair = amp_tom_fit(
                x=amplitudes,
                y=num_terms[t],
                q_pair=pair_key,
                term=t,
                result_dict=fit_params_pair,
            )

        amp_lin_fit_params[pair_key] = fit_params_pair
        target_amplitudes = estimate_cancellation_amplitudes(
            amplitudes=amplitudes, ham_term=num_terms, ampl_params=fit_params_pair
        )
        cancellating_amplitudes[pair_key] = target_amplitudes

    return amp_hamiltonian_params, amp_lin_fit_params, cancellating_amplitudes


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
        result_dict[term] = {
            "a": popt[0],
            "b": popt[1],
            "omega": popt[2],
            "phi": popt[3],
        }
    except Exception as e:
        log.warning(f"{term} term vs amplitudes fit failed for {q_pair} due to {e}.")

    return result_dict


def phase_tomography_cr_fit(data: Data):
    phase_hamiltonian_params = {}
    for phase in data.phases:
        phase_data = data.select_phase(phase)
        phase_length_params, cr_duration = tomography_cr_fit(phase_data)
        for pair in phase_data.pairs:
            terms = extract_hamiltonian_terms(pair, phase_length_params)
            terms = refactor_hamiltoanian_terms(terms, pair)
            res_tuple = (phase, terms)
            if pair not in phase_hamiltonian_params:
                phase_hamiltonian_params[pair] = [res_tuple]
            else:
                phase_hamiltonian_params[pair].append(res_tuple)

    phase_sin_fit_params = {}
    cancellating_phases = {}
    for pair_key, pair_value in phase_hamiltonian_params.items():
        num_terms = {t: [] for t in HamiltonianTerm}
        phases = []
        for vals in pair_value:
            phases.append(vals[0])
            for t in HamiltonianTerm:
                num_terms[t].append(vals[1][t])

        fit_params_pair = {}
        for t in HamiltonianTerm:
            fit_params_pair = phase_tom_fit(
                x=phases,
                y=num_terms[t],
                q_pair=pair_key,
                term=t,
                result_dict=fit_params_pair,
            )

        phase_sin_fit_params[pair_key] = fit_params_pair
        ctrl_phase, trgt_phase = estimate_cr_phases(
            phases=phases, ham_term=num_terms, phase_params=fit_params_pair
        )
        cancellating_phases[pair_key] = {"control": ctrl_phase, "target": trgt_phase}

    return phase_hamiltonian_params, phase_sin_fit_params, cancellating_phases


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
                            gamma=fit.fitted_parameters[target[0], target[1], setup][3],
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

                fit_dict = {}
                if type(fit).__name__ == "HamiltonianTomographyCRLengthResults":
                    fit_dict = fit.cr_lengths
                    annotation = "CR gate duration [ns]"

                if type(fit).__name__ == "HamiltonianTomographyCRAmplitudeResults":
                    fit_dict = fit.cr_amplitudes
                    annotation = "CR gate amplitude [a.u.]"

                if target in fit_dict:
                    fig.add_vline(
                        x=fit_dict[target],
                        line_width=2,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=annotation,
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
    if type(fit).__name__ == "HamiltonianTomographyCRLengthResults":
        fit_dict = fit.cr_lengths
        annotation = "CR gate duration [ns]"

    if type(fit).__name__ == "HamiltonianTomographyCRAmplitudeResults":
        fit_dict = fit.cr_amplitudes
        annotation = "CR gate amplitude [a.u.]"

    if target in fit_dict:
        fig.add_vline(
            x=fit_dict[target],
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=annotation,
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
        "HamiltonianTomographyCANCAmplData",  # noqa: F821
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

    if type(data).__name__ == "HamiltonianTomographyCANCAmplData":
        fit_func = fitting.linear_fit
        x_title = "amplitude [a.u.]"
        tunable_params = fit.cancellation_pulse_amplitudes[target]
        plotting_terms = {
            HamiltonianTerm.IX: "ampl_ix",
            HamiltonianTerm.IY: "ampl_iy",
        }

    if type(data).__name__ == "HamiltonianTomographyCRPhaseData":
        fit_func = fitting.sin_fit
        x_title = "phase [rad.]"
        tunable_params = {}
        tunable_params["phi0"] = fit.cancellation_pulse_phases[target]["control"]
        tunable_params["phi1"] = (
            fit.cancellation_pulse_phases[target]["control"]
            - fit.cancellation_pulse_phases[target]["target"]
        )
        plotting_terms = {
            HamiltonianTerm.ZY: "phi0",
            HamiltonianTerm.IY: "phi1",
        }

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
                        line=go.scatter.Line(dash="dot"),
                    )
                ]
            )

            if target in fit.fitted_parameters and t in fit.fitted_parameters[target]:
                amp_range = np.linspace(
                    min(exp_sweeper),
                    max(exp_sweeper),
                    2 * len(exp_sweeper),
                )
                params = fit.fitted_parameters[target][t]
                fig.add_trace(
                    go.Scatter(
                        x=amp_range,
                        y=fit_func(amp_range, **params),
                        name=f"{t.name} Fit",
                        mode="lines",
                    ),
                )

                if t in plotting_terms:
                    params_name = plotting_terms[t]
                    fig.add_vline(
                        x=tunable_params[params_name],
                        name=f"{params_name}",
                        line_dash="dash",
                    )

                fig.update_layout(
                    showlegend=True,
                    xaxis_title=(f"{x_title}"),
                    yaxis_title="Interaction strength [MHz]",
                )

                fitting_report = table_html(
                    table_dict(
                        8 * [target],
                        (
                            [
                                f"{term.name}: Fitted_parameters"
                                for term in HamiltonianTerm
                            ]
                            + [k for k in tunable_params.keys()]
                        ),
                        (
                            [
                                fit.fitted_parameters[target][term]
                                for term in HamiltonianTerm
                            ]
                            + [v for v in tunable_params.values()]
                        ),
                    )
                )

            figures.append(fig)

    return figures, fitting_report
