"""Hamiltonian tomography protocol for CR gate calibration.

This protocol computes the expectation values for X, Y and Z for the target qubit
after the application of a cross resonance sequence. The CR pulses are played on the control drive
channel with frequency set to the frequency of the target drive channel.
"""

from dataclasses import dataclass, field

import numpy as np
from qibolab import (
    AcquisitionType,
    AveragingMode,
    ParallelSweepers,
    Parameter,
    Sweeper,
)
from scipy.constants import kilo

from qibocal.auto.operation import (
    Protocol,
    QubitId,
    QubitPairId,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import table_dict, table_html

from .cr_parent_classes import (
    Basis,
    HamiltonianTerm,
    HamiltonianTomographyData,
    HamiltonianTomographyParameters,
    HamiltonianTomographyResults,
    SetControl,
)
from .cross_resonance_processing import (
    cancellation_amplitude_fit,
    reconstruct_full_hamiltonian_terms,
)
from .plotting import cancellation_calibration_plot
from .utils import (
    cross_resonance_experiment,
    cross_resonance_pulses,
    ro_delay_range,
    update_cnot_from_fit,
)

__all__ = ["cancellation_amplitude_tuning"]

HamiltonianTomographyCANCAmplType = np.dtype(
    [
        ("prob_target", np.float64),
        ("error_target", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
        ("amp", np.float64),
        ("x", np.float64),
    ]
)
"""Custom dtype for Cancellation amplitude."""


@dataclass(kw_only=True)
class HamiltonianTomographyCANCAmplParameters(HamiltonianTomographyParameters):
    """HamiltonianTomographyCANCAmplitude runcard inputs."""

    target_ampl_range: tuple[float, float, float]
    """Amplitude range of cancellation pulse."""
    verbose_plot: bool = False
    """If `True` in the report all the single Hamiltonian tomographies are plotted."""


@dataclass(kw_only=True)
class HamiltonianTomographyCANCAmplResults(HamiltonianTomographyResults):
    """HamiltonianTomographyCANCAmpl outputs."""

    hamiltonian_terms: dict[
        tuple[QubitId, QubitId], list[tuple[float, dict[HamiltonianTerm, float]]]
    ] = field(default_factory=dict)
    """Terms in effective Hamiltonian."""

    fitted_parameters: dict[tuple[QubitId, QubitId], dict[HamiltonianTerm, list]] = (
        field(default_factory=dict)
    )
    """Fitted parameters for Hamiltonian Terms values for different amplitudes."""

    cancellation_pulse_amplitudes: dict[QubitPairId, dict[str, float]] = field(
        default_factory=dict
    )
    """Fitted parameters for cancellation pulse amplitudes."""

    hamiltonian_tom_params: dict[
        float, dict[tuple[QubitId, QubitId, SetControl], list[float]]
    ] = field(default_factory=dict)
    """Fitted parameters for of Hamiltonian Tomography experiment for each qubit and per amplitude value.
    Used for plotting with the `verbose_plot` option"""

    cr_lengths: dict[float, dict[tuple[QubitId, QubitId, SetControl], list[float]]] = (
        field(default_factory=dict)
    )
    """Cross resonance pulse duration found in Hamiltonian Tomography experiment for each qubit and per amplitude value.
    Used for plotting with the `verbose_plot` option"""

    verbose_plot: bool = False
    """If `True` in the report all the single Hamiltonian tomographies are plotted."""

    def select_pair_and_ampl_ham_params(self, amplitude: float, pair: QubitPairId):
        """Data
        Select and refactor Hamiltonian tomography parameters for a given amplitude and qubit pair.
        """
        selected_ham_tom_params = next(
            (
                val_ham_params
                for (amp, val_ham_params) in self.hamiltonian_terms[pair]
                if amp == amplitude
            ),
            None,
        )

        return reconstruct_full_hamiltonian_terms(selected_ham_tom_params, pair)


@dataclass(kw_only=True)
class HamiltonianTomographyCANCAmplData(HamiltonianTomographyData):
    """Data structure for CANC Amplitude."""

    verbose_plot: bool = False
    """If `True` in the report all the single Hamiltonian tomographies are plotted."""

    @property
    def amplitudes(self) -> list[float]:
        first_key = next(iter(self.data.keys()))
        return np.unique(self.data[first_key].amp).tolist()

    def select_amplitude(self, amplitude: float) -> HamiltonianTomographyData:
        new_data = HamiltonianTomographyData(echo=self.echo)
        new_data.data = {k: d[d.amp == amplitude] for k, d in self.data.items()}
        return new_data

    def register_qubit(self, dtype, data_keys, data_dict):
        """Store output for single qubit."""
        duration_list = data_dict["x"]
        amp_list = data_dict["amp"]
        size = len(duration_list) * len(amp_list)
        ar = np.empty(size, dtype=dtype)
        amplitudes, durations = np.meshgrid(amp_list, duration_list)
        ar["x"] = durations.ravel()
        ar["amp"] = amplitudes.ravel()
        ar["prob_target"] = data_dict["prob_target"].ravel()
        ar["error_target"] = data_dict["error_target"].ravel()
        ar["prob_control"] = data_dict["prob_control"].ravel()
        ar["error_control"] = data_dict["error_control"].ravel()

        self.data[data_keys] = np.rec.array(ar)


def _acquisition(
    params: HamiltonianTomographyCANCAmplParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCANCAmplData:
    """Data acquisition for Hamiltonian tomography CR protocol.

    We measure the expectation values X,Y and Z on the target qubit after
    applying the CR sequence specified by the input parameters. We repeat the
    measurement twice for each target qubit, once with the control qubit in state 0
    and once with the control qubit in state 1.

    We store the probability of the control qubit and the expectation value of the target qubit.

    """

    data = HamiltonianTomographyCANCAmplData(
        echo=params.echo,
        verbose_plot=params.verbose_plot,
    )

    updates = []
    control_ampls: dict[QubitPairId, float] = {}
    control_phases: dict[QubitPairId, float] = {}
    target_phases: dict[QubitPairId, float] = {}
    for pair in targets:
        control, target = pair

        updates.append(
            {
                platform.qubits[control].drive_extra[target]: {
                    "frequency": platform.config(
                        platform.qubits[target].drive
                    ).frequency
                }
            }
        )

        cr_pulse, canc_pulse = cross_resonance_pulses(platform, control, target)
        control_ampls |= {pair: cr_pulse.amplitude}
        control_phases |= {pair: cr_pulse.relative_phase}
        target_phases |= {pair: canc_pulse.relative_phase}

    for basis in Basis:
        for setup in SetControl:
            sequence, cr_pulses, cr_target_pulses, cr_delays, ro_delays = (
                cross_resonance_experiment(
                    platform=platform,
                    pair_list=targets,
                    duration=0.0,
                    ctrl_ampl=control_ampls,
                    ctrl_phase=control_phases,
                    targ_ampl=0.0,
                    targ_phase=target_phases,
                    basis=basis,
                    setup=setup,
                    echo=params.echo,
                    interpolated_sweeper=params.interpolated_sweeper,
                )
            )

            if params.interpolated_sweeper:
                duration_parallel_sweeper = ParallelSweepers(
                    [
                        Sweeper(
                            parameter=Parameter.duration_interpolated,
                            range=params.duration_range,
                            pulses=cr_pulses[pair] + cr_target_pulses[pair],
                        )
                        for pair in targets
                    ]
                )
            else:
                duration_parallel_sweeper = ParallelSweepers(
                    [
                        Sweeper(
                            parameter=Parameter.duration,
                            range=params.duration_range,
                            pulses=cr_pulses[pair]
                            + cr_target_pulses[pair]
                            + cr_delays[pair],
                        )
                        for pair in targets
                    ]
                )
                ro_sweeper = ParallelSweepers(
                    [
                        Sweeper(
                            parameter=Parameter.duration,
                            range=ro_delay_range(
                                cr_pulse_duration_range=params.duration_range,
                                echo=params.echo,
                                cntl_setup=setup,
                                control=c,
                                platform=platform,
                            ),
                            pulses=ro_delays[(c, t)],
                        )
                        for c, t in targets
                    ]
                )
                duration_parallel_sweeper += ro_sweeper

            amp_sweepers = ParallelSweepers(
                [
                    Sweeper(
                        parameter=Parameter.amplitude,
                        range=params.target_ampl_range,
                        pulses=[cr_target_pulses[pair][0] for pair in targets],
                    )
                ]
            )
            if params.echo:
                # sweeping over the out-of-phase signal (refocusing) of the cancellation pulse
                echo_ampl_range = tuple(-x for x in params.target_ampl_range)
                amp_sweepers += ParallelSweepers(
                    [
                        Sweeper(
                            parameter=Parameter.amplitude,
                            range=echo_ampl_range,
                            pulses=[cr_target_pulses[pair][1] for pair in targets],
                        )
                    ]
                )

            results = platform.execute(
                [sequence],
                [
                    duration_parallel_sweeper,
                    amp_sweepers,
                ],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.CYCLIC,
                updates=updates,
            )

            for ctrl, targ in targets:
                target_acq_handle = list(
                    sequence.channel(platform.qubits[targ].acquisition)
                )[-1].id
                control_acq_handle = list(
                    sequence.channel(platform.qubits[ctrl].acquisition)
                )[-1].id

                prob_target = results[target_acq_handle].ravel()
                prob_control = results[control_acq_handle].ravel()

                data.register_qubit(
                    HamiltonianTomographyCANCAmplType,
                    (ctrl, targ, basis, setup),
                    dict(
                        x=np.arange(*params.duration_range),
                        amp=np.arange(*params.target_ampl_range),
                        prob_target=1 - 2 * prob_target,
                        error_target=2
                        * np.sqrt(prob_target * (1 - prob_target) / params.nshots),
                        prob_control=1 - 2 * prob_control,
                        error_control=2
                        * np.sqrt(prob_control * (1 - prob_control) / params.nshots),
                    ),
                )

    return data


def _fit(
    data: HamiltonianTomographyCANCAmplData,
) -> HamiltonianTomographyCANCAmplResults:
    """Post-processing function for HamiltonianTomographyCANCAmpl.

    We fit the expectation values using the Eq. S10 from the paper https://arxiv.org/pdf/2303.01427.
    Afterwards, we extract the Hamiltonian terms from the fitted parameters.

    """
    hamiltonian_terms, fitted_parameters, cal_amplitudes, ham_tom_params, cr_lengths = (
        cancellation_amplitude_fit(
            data=data,
        )
    )

    return HamiltonianTomographyCANCAmplResults(
        echo=data.echo,
        hamiltonian_terms=hamiltonian_terms,
        fitted_parameters=fitted_parameters,
        cancellation_pulse_amplitudes=cal_amplitudes,
        hamiltonian_tom_params=ham_tom_params,
        cr_lengths=cr_lengths,
        verbose_plot=data.verbose_plot,
    )


def _plot(
    data: HamiltonianTomographyCANCAmplData,
    target: QubitPairId,
    fit: HamiltonianTomographyCANCAmplResults,
):
    """Plotting function for HamiltonianTomographyCANCAmpl."""
    figs, fitting_report = cancellation_calibration_plot(data, target, fit)

    plot_ham_tom = True if fit is None else fit.verbose_plot

    if plot_ham_tom:
        from .plotting import (
            tomography_cr_plot,
        )

        for a in data.amplitudes:
            ampl_data = data.select_amplitude(a)
            if fit is not None:
                selected_ham_terms = fit.select_pair_and_ampl_ham_params(a, target)
                ham_tom_fit = HamiltonianTomographyResults(
                    echo=fit.echo,
                    hamiltonian_terms=selected_ham_terms,
                    fitted_parameters=fit.hamiltonian_tom_params[a],
                    cr_lengths=fit.cr_lengths[a],
                )

                fitting_report += "\n" + table_html(
                    table_dict(
                        8 * [target],
                        [f"{term.name} [MHz]" for term in HamiltonianTerm]
                        + ["CR duration (ns)", "Control amplitude [a.u.]"],
                        [
                            ham_tom_fit.hamiltonian_terms[target[0], target[1], term]
                            * kilo
                            for term in HamiltonianTerm
                        ]
                        + [
                            fit.cr_lengths[a][target]
                            if target in fit.cr_lengths[a]
                            else None
                        ]
                        + [a],
                    )
                )
            else:
                ham_tom_fit = None
                fitting_report = ""

            f, _ = tomography_cr_plot(ampl_data, target, ham_tom_fit)
            figs += f

    return figs, fitting_report


def _update(
    results: HamiltonianTomographyCANCAmplResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):

    # now no check is needed since the acquisition was executed correctly,
    # which means we have all parameters defined.
    cr_pulse, canc_pulse = cross_resonance_pulses(platform, target[0], target[1])
    gate_duration = cr_pulse.duration
    control_amplitude = cr_pulse.amplitude
    control_phase = cr_pulse.relative_phase
    target_phase = canc_pulse.relative_phase

    # check if the resulting fit was succsessfull
    if target in results.cancellation_pulse_amplitudes:
        update_cnot_from_fit(
            platform=platform,
            pair=target,
            cr_duration=gate_duration,
            cr_ampl=control_amplitude,
            control_phase=control_phase,
            canc_ampl=results.cancellation_pulse_amplitudes[target]["ampl_iy"],
            canc_phase=target_phase,
            echo_flag=results.echo,
        )


cancellation_amplitude_tuning = Protocol(_acquisition, _fit, _plot, _update)
"""HamiltonianTomographyCANCAmpl Protocol object."""
