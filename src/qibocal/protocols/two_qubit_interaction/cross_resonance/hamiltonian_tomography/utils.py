from enum import Enum
from typing import Callable, Optional, Union

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
from ....utils import angle_wrap, quinn_fernandes_algorithm, table_dict, table_html
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


def dynamic_evolution_optimizer(
    signals_id: np.ndarray,
    x: np.ndarray,
    init_omega_guess: float,
    use_constraints: bool = False,
) -> np.ndarray:
    """Optimizer for sinusoidal fitting; it exploits evolution algorithm to find the best fit parameters.
    This algorithm is a gradient-free optimization, hence might be less affected by local minima in the cost
    function landscape, but it can be computationally expensive, especially for large datasets or complex models.
    """

    assert x.ndim == 1, f"Expected 1D array, got array with shape {x.shape}"
    concat_x = np.concatenate([x, x, x])

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
    concatenated_signal: np.ndarray,
    vector_x: np.ndarray,
    init_omega_guess: float,
) -> np.ndarray:
    """Optimizer for sinusoidal fitting; it exploits gradient-based algorithms to find the best fit parameters.
    This algorithm is a gradient-base optimization, hence might be affected by local minima in the cost
    function landscape when dealing with sinusoidal functions due to their periodicity.
    """

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
    )

    return popt


def numerical_root_finder(
    root_func: Callable[..., float],
    x_range: np.ndarray | list[float | int],
    tol: float,
    **kwargs,
):
    """Finds the root of a generic function :data:`root_func` by minimizing its absolute value.

    The solver first performs a coarse grid search across the provided range to
    identify a starting candidate, then refines the result using the
    Nelder-Mead simplex algorithm; then, if the search is successful, the
    algorithm returns the value where the function is closest to zero.
    This function minimizes `abs(root_func(x))`. If the function does not
    cross zero, it will return the local minimum of the absolute value.
    """

    def func_to_solve(x):
        return np.abs(root_func(x, **kwargs))

    x_grid = np.linspace(x_range[0], x_range[-1], 100 * len(x_range))

    y_vals = np.array([func_to_solve(xi) for xi in x_grid]).flatten()

    cr_sig = y_vals[y_vals - np.min(y_vals) <= tol][0]

    # Use a numerical minimizer starting from our guess
    res = minimize(
        func_to_solve,
        x0=cr_sig,
        method="Nelder-Mead",
    )

    if res.success:
        cr_sig = min(res.x)

    return cr_sig if cr_sig <= np.max(x_grid) else np.max(x_grid)


def compute_total_expectation_value(
    data: Union[
        "HamiltonianTomographyCRLengthData",  # noqa: F821
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
    ],
    pair: QubitPairId,
) -> np.ndarray:
    """Given a Qubit Pair :data:`pair`=(control, target), it computes the expectation
    values for each Pauli Basis for the target qubit.
    """

    tot_exp_vals = []
    for basis in Basis:
        tot_exp_vals.append(
            data.data[pair[0], pair[1], basis, SetControl.Id].prob_target
            + data.data[pair[0], pair[1], basis, SetControl.X].prob_target
        )
    return np.vstack(tot_exp_vals)


def bloch_func(
    x: list[float | int] | np.ndarray, pair: QubitPairId, fitted_parameters: dict
) -> np.ndarray:
    """Given the fitted parameters for the target's Pauli expectation values
    for either control qubit in |0> or in |1>, computes the estimated Bloch vector.
    """

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
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
    ],
    pair: QubitPairId,
    fitted_parameters: dict | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """For a given qubit pair :data:`pair`, it computes the Bloch vector R for each data point and
    also estimates the Bloch vector using the fitted parameters of the Hamiltonian Tomography.
    See `arXiv:1603.04821 <https://arxiv.org/abs/1603.04821>`__ for further information.
    """

    bloch_exp = compute_total_expectation_value(data, pair)
    bloch_exp = np.sqrt(np.sum((bloch_exp) ** 2, axis=0))

    bloch_fit = None
    if fitted_parameters is not None:
        times = data.data[pair[0], pair[1], Basis.Z, SetControl.Id].x
        times_range = np.linspace(min(times), max(times), 2 * len(times))

        bloch_fit = bloch_func(times_range, pair, fitted_parameters)

    return bloch_exp, bloch_fit


def estimate_cr_param(
    x_range: np.ndarray,
    data: "HamiltonianTomographyCRLengthData",  # noqa: F821
    pair: QubitPairId,
    fitted_parameters: dict,
    tol: float = 1e-6,
) -> float | int:
    """Function for estimating important parameters for the cross resonance, depending on the
    specific experiment run; if :data:`data` is type :class:`HamiltonianTomographyData` it finds the
    thuned pulse duration, while if :data:`data` is type :class:`HamiltonianTomographyCRAmplitudeData` it
    finds the tuned pulse amplitude.

    The cross resonance parameter is computed by finding the value that solves the Bloch vector R.
    To do so, the function checks whether all the fits in the Hamiltonian Tomography experiment succeeded;
    if so the Bloch vector is computed by using the fitted parameter, otherwise it simply computes
    R only for the acquired datapoints.
    """

    if all([(pair[0], pair[1], s) in fitted_parameters for s in SetControl]):
        bloch_data, _ = compute_bloch_vector(data, pair, fitted_parameters)
        x_range = data.data[pair[0], pair[1], Basis.Z, SetControl.Id].x
        param = numerical_root_finder(
            root_func=bloch_func,
            x_range=x_range,
            tol=tol,
            pair=pair,
            fitted_parameters=fitted_parameters,
        )
    else:
        bloch_data, _ = compute_bloch_vector(data, pair)
        idx = np.argmin(bloch_data)
        param = x_range[idx]

    if data.__name__ == "HamiltonianTomographyCRLengthData":
        # time duration must be integer
        return int(param)

    return float(param)


def tune_cancellation_sequence(
    x: list[float | int] | np.ndarray,
    function_to_tune: Callable[..., float],
    interactions_to_analyze: list[HamiltonianTerm],
    ham_term: dict,
    fit_params: dict,
    tuned_keys: list[str],
    tol: float,
) -> dict[str, float]:
    """Function for estimating parameters for the cancellation pulse sequence, depending on the
    specific experiment (either pulses phase tuning or cancellation amplitude tuning).

    The specific parameter is computed by finding the value that solves an input function :data:`function_to_tune`.
    To do so, the function checks whether all the fits in the Hamiltonian Tomography experiment succeeded;
    if so roots are solved by using the fitted parameter, otherwise it simply computes :data:`function_to_tune` only
    for the acquired datapoints and the closest value to 0 is selected.

    These parameters are computed for every term of the Cross Resonance Hamiltonian: IX, ZX, IY, ZY, IZ, ZZ and saved
    into a dictionary.
    """

    assert len(tuned_keys) == len(interactions_to_analyze), (
        """tuned_keys and interactions_to_analyze must be equally long."""
    )

    # converting list into numpy array
    x = np.array(x)

    tuned_parameters = {}
    for ham_int, k in zip(interactions_to_analyze, tuned_keys):
        if ham_int in fit_params:
            tuned_parameters[k] = float(
                numerical_root_finder(
                    root_func=function_to_tune,
                    x_range=x,
                    tol=tol,
                    **fit_params[ham_int],
                )
            )
        else:
            selected_ham_term = np.abs(np.array(ham_term[ham_int]))
            min_idx = np.argmin(selected_ham_term)
            tuned_parameters[k] = float(x[min_idx])

    return tuned_parameters


def estimate_cancellation_amplitudes(
    amplitudes: list[float | int] | np.ndarray,
    ham_term: dict,
    ampl_params: dict,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Extrapolates the cancellation pulse amplitude for the cross resonance pulse sequence.

    This function estimates the optimal amplitudes for cancelling unwanted Hamiltonian terms
    (IX and IY) in the cross resonance pulse sequence. It uses linear fitting to determine
    the amplitude values that minimize these interaction terms.
    """

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
    phases: list[float | int] | np.ndarray,
    ham_term: dict,
    phase_params: dict,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Extrapolates the cancellation pulse phases for the cross resonance pulse sequence.

    This function estimates the optimal phases for cancelling unwanted Hamiltonian terms
    (ZY and IY) in the cross resonance pulse sequence. It uses sinusoidal fitting to determine
    the phase values that minimize these interaction terms.
    """

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
        tuned_phases["phi0"] += phase_params[HamiltonianTerm.ZY]["omega"] * np.pi

    if fitting.sin_fit(tuned_phases["phi1"], **phase_params[HamiltonianTerm.IX]) > 0:
        # same as above, but this time is more euristic.
        tuned_phases["phi1"] += phase_params[HamiltonianTerm.IY]["omega"] * np.pi

    return angle_wrap(tuned_phases["phi0"]), angle_wrap(
        tuned_phases["phi0"] - tuned_phases["phi1"]
    )


def tomography_cr_fit(
    data: Union[
        "HamiltonianTomographyCRLengthData",  # noqa: F821
        "HamiltonianTomographyCRAmplitudeData",  # noqa: F821
    ],
    fit_with_evolution: bool = False,
) -> tuple[
    dict[tuple[QubitId, QubitId, SetControl], list[float]],
    dict[tuple[QubitId, QubitId], int],
]:
    """Fit Hamiltonian tomography data for cross-resonance gates.

    This function performs sinusoidal fitting on tomography data collected for
    cross-resonance interactions between qubit pairs. It fits the measurement
    probabilities in X, Y, and Z bases across different control settings.
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

            total_omega_guess = quinn_fernandes_algorithm(
                concatenated_signal, vector_x, sampling_rate, speedup_flag=True
            )

            if fit_with_evolution:
                popt = dynamic_evolution_optimizer(
                    concatenated_signal,
                    vector_x,
                    total_omega_guess,
                )
            else:
                popt = scipy_curve_fit_optimizer(
                    concatenated_signal,
                    vector_x,
                    total_omega_guess,
                )

            fitted_parameters[pair[0], pair[1], setup] = popt.tolist()

        cr_gate_x[pair[0], pair[1]] = estimate_cr_param(
            vector_x, data, pair, fitted_parameters
        )

    return fitted_parameters, cr_gate_x


def extract_hamiltonian_terms(
    pair: QubitPairId,
    fitted_parameters: dict[tuple[QubitId, QubitId, SetControl], list[float]],
) -> dict[QubitPairId, dict[HamiltonianTerm, float]]:
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


def refactor_hamiltonian_terms(
    ham_terms: dict[tuple[QubitId, QubitId, HamiltonianTerm], float],
    pair: QubitPairId,
) -> dict[HamiltonianTerm, float]:
    """Refactor Hamiltonian terms by removing qubit pair information from keys.

    Converts dictionary keys from (qubit_id_0, qubit_id_1, HamiltonianTerm) tuples
    to just HamiltonianTerm, simplifying the data structure.
    """

    for term in HamiltonianTerm:
        ham_terms[term] = ham_terms.pop((pair[0], pair[1], term))

    return ham_terms


def reconstruct_full_hamiltonian_terms(
    ham_terms: dict[HamiltonianTerm, float],
    pair: QubitPairId,
) -> dict[tuple[QubitId, QubitId, HamiltonianTerm], float]:
    """
    Reconstruct full Hamiltonian terms dictionary by adding qubit pair information to keys.

    Converts dictionary keys from HamiltonianTerm to (qubit_id_0, qubit_id_1, HamiltonianTerm) tuples,
    restoring the original data structure used for Hamiltonian terms.
    """
    for term in HamiltonianTerm:
        ham_terms[(pair[0], pair[1], term)] = ham_terms.pop(term)

    return ham_terms


def amp_tom_fit(
    x: list[float | int] | np.ndarray,
    y: list[float | int] | np.ndarray,
    q_pair: QubitPairId,
    term: HamiltonianTerm,
    result_dict: dict[HamiltonianTerm, dict[str, float]],
) -> dict[HamiltonianTerm, dict[str, float]]:
    """Fit linear function to amplitude vs Hamiltonian term data.

    Performs a linear fit on the provided data to extract amplitude-dependent
    parameters for a specific Hamiltonian term.
    """
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


def cancellation_amplitude_fit(
    data: Data,
) -> tuple[
    dict[QubitPairId, list[tuple[float, dict[HamiltonianTerm, float]]]],
    dict[QubitPairId, dict[HamiltonianTerm, dict[str, float]]],
    dict[QubitPairId, dict[str, float]],
    dict[float, dict[tuple[QubitId, QubitId, SetControl], list[float]]],
    dict[float, dict[tuple[QubitId, QubitId], int]],
]:
    """Perform amplitude-dependent tomography fitting for calibrating
    cross resonance cancellation pulse.

    Fits the dependence of Hamiltonian term parameters on the CR pulse amplitude.
    Extracts Hamiltonian terms at different amplitudes and fits their variation
    with amplitude to obtain linear parameters and cancellation amplitudes.
    """

    amp_hamiltonian_params = {}
    ham_tomography_dict = {}
    gate_duration_dict = {}
    for amp in data.amplitudes:
        amp_data = data.select_amplitude(amp)
        length_params, cr_durations = tomography_cr_fit(amp_data)
        gate_duration_dict[amp] = cr_durations
        ham_tomography_dict[amp] = length_params
        for pair in amp_data.pairs:
            terms = extract_hamiltonian_terms(pair, length_params)
            terms = refactor_hamiltonian_terms(terms, pair)
            res_tuple = (amp, terms)
            if pair not in amp_hamiltonian_params:
                amp_hamiltonian_params[pair] = [res_tuple]
            else:
                amp_hamiltonian_params[pair].append(res_tuple)

    amp_lin_fit_params = {}
    calibrated_amplitudes = {}
    for pair_key, pair_value in amp_hamiltonian_params.items():
        num_terms = {t: [] for t in HamiltonianTerm}
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
        calibrated_amplitudes[pair_key] = target_amplitudes

    return (
        amp_hamiltonian_params,
        amp_lin_fit_params,
        calibrated_amplitudes,
        ham_tomography_dict,
        gate_duration_dict,
    )


def phase_tom_fit(
    x: list[float | int] | np.ndarray,
    y: list[float | int] | np.ndarray,
    q_pair: QubitPairId,
    term: HamiltonianTerm,
    result_dict: dict[HamiltonianTerm, dict[str, float]],
) -> dict[HamiltonianTerm, dict[str, float]]:
    """Fit sinusoidal function to phase vs Hamiltonian term data.

    Performs a sinusoidal fit on the provided data to extract phase-dependent
    parameters for a specific Hamiltonian term.
    """

    median_sig = np.median(y)
    q80 = np.quantile(y, 0.8)
    q20 = np.quantile(y, 0.2)
    amplitude_guess = abs(q80 - q20) / QUANTILE_CONSTANT
    phase_guess = 0
    pguess = [amplitude_guess, median_sig, phase_guess]
    try:
        popt, _ = curve_fit(
            lambda x, a, b, phi: fitting.sin_fit(x, a, b, 1, phi),
            x,
            y,
            p0=pguess,
            maxfev=int(1e6),
            absolute_sigma=True,
            bounds=(
                [-np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf],
            ),
        )
        result_dict[term] = {
            "a": popt[0],
            "b": popt[1],
            "omega": 1,
            "phi": popt[2],
        }
    except Exception as e:
        log.warning(f"{term} term vs amplitudes fit failed for {q_pair} due to {e}.")

    return result_dict


def cancellation_phase_fit(
    data: Data,
) -> tuple[
    dict[QubitPairId, list[tuple[float, dict[HamiltonianTerm, float]]]],
    dict[QubitPairId, dict[HamiltonianTerm, dict[str, float]]],
    dict[QubitPairId, dict[str, float]],
    dict[float, dict[tuple[QubitId, QubitId, SetControl], list[float]]],
    dict[float, dict[tuple[QubitId, QubitId], int]],
]:
    """Fit phase-dependent Hamiltonian parameters using cross-resonance tomography.

    Extracts and fits Hamiltonian terms for different phases across all qubit pairs,
    performing sinusoidal fits on phase-dependent data and estimating cancelling phases
    for control and target qubits.
    """

    phase_hamiltonian_params = {}
    ham_tomography_dict = {}
    gate_duration_dict = {}
    for p in data.phases:
        phase_data = data.select_phase(p)
        length_params, cr_durations = tomography_cr_fit(phase_data)
        gate_duration_dict[p] = cr_durations
        ham_tomography_dict[p] = length_params
        for pair in phase_data.pairs:
            terms = extract_hamiltonian_terms(pair, length_params)
            terms = refactor_hamiltonian_terms(terms, pair)
            res_tuple = (p, terms)
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

    return (
        phase_hamiltonian_params,
        phase_sin_fit_params,
        cancellating_phases,
        ham_tomography_dict,
        gate_duration_dict,
    )


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
    """Generate plots for Hamiltonian tomography experiment."""

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


def cancellation_calibration_plot(
    data: Union[
        "HamiltonianTomographyCANCAmplData",  # noqa: F821
        "HamiltonianTomographyCANCPhaseData",  # noqa: F821
    ],
    target: QubitPairId,
    fit: Optional[
        Union[
            "HamiltonianTomographyCANCAmplitudeResults",  # noqa: F821
            "HamiltonianTomographyCANCPhaseResults",  # noqa: F821
        ]
    ] = None,
) -> tuple[list[go.Figure], str]:
    """Plot calibration results for cross-resonance Hamiltonian tomography when
    tuning cancellation pulses.

    Generates plots for either amplitude or phase calibration data of cancellation pulses
    in cross-resonance interactions. Fits effective Hamiltonian terms and visualizes the
    results with fitted curves overlaid on experimental data.
    """

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

    if type(data).__name__ == "HamiltonianTomographyCANCPhaseData":
        fit_func = fitting.sin_fit
        x_title = "phase [rad.]"
        tunable_params = {}
        tunable_params["phi0"] = fit.cancellation_pulse_phases[target]["control"]
        tunable_params["phi1"] = angle_wrap(
            fit.cancellation_pulse_phases[target]["control"]
            - fit.cancellation_pulse_phases[target]["target"]
        )
        plotting_terms = {
            HamiltonianTerm.ZY: "phi0",
            HamiltonianTerm.IY: "phi1",
        }

    if fit is not None:
        for t in HamiltonianTerm:
            eff_ham_term = []
            exp_sweeper = []
            for f in fit.hamiltonian_terms[target]:
                eff_ham_term.append(f[1][t])
                exp_sweeper.append(f[0])
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
                    2 * len(exp_sweeper) if len(exp_sweeper) >= 100 else 200,
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
