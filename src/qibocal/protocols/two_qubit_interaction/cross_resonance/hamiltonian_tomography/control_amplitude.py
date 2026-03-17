"""Hamiltonian tomography protocol for CR gate calibration.

This protocol computes the expectation values for X, Y and Z for the target qubit
after the application of a cross resonance sequence. The CR pulses are played on the control drive
channel with frequency set to the frequency of the target drive channel.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Sweeper,
)
from scipy.constants import kilo

from .....auto.operation import (
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from .....calibration import CalibrationPlatform
from ....utils import table_dict, table_html
from ..utils import Basis, SetControl, cr_sequence
from .utils import (
    EPS,
    HamiltonianTerm,
    HamiltonianTomographyData,
    amplitude_tomography_cr_fit,
    calibration_cr_plot,
    reconstruct_full_hamiltonian_terms,
)

HamiltonianTomographyCRAmplType = np.dtype(
    [
        ("prob_target", np.float64),
        ("error_target", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
        ("amp", np.float64),
        ("x", np.int64),
    ]
)
"""Custom dtype for Cancellation amplitude."""


@dataclass
class HamiltonianTomographyCRAmplParameters(Parameters):
    """HamiltonianTomographyCRAmplitude runcard inputs."""

    pulse_duration_start: float
    """Initial duration of CR pulse [ns]."""
    pulse_duration_end: float
    """Final duration of CR pulse [ns]."""
    pulse_duration_step: float
    """Step CR pulse duration [ns]."""
    control_ampl_start: float
    """Amplitude of cancellation pulse."""
    control_ampl_end: float
    """Final amplitude of CR pulse."""
    control_ampl_step: float
    """Step CR pulse amplitude."""
    control_phase: float | None = 0.0
    """Phase of the CR pulse on the control qubit"""
    target_phase: float | None = 0.0
    """Phase of the Cancellation pulse on the target qubit"""
    target_amplitude: float | None = 0.0
    """Amplitude of the Cancellation pulse on the target qubit"""
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""
    echo: bool = False
    """Apply echo sequence or not.

    The ECR is described in https://arxiv.org/pdf/1210.7011
    """
    verbose_plot: bool = False
    """If `True` in the report all the single Hamiltonian tomographies are plotted."""

    @property
    def amplitude_range(self) -> np.ndarray:
        """Amplitude range for CR pulses."""
        return np.arange(
            self.control_ampl_start,
            self.control_ampl_end,
            self.control_ampl_step,
        )

    @property
    def duration_range(self) -> np.ndarray:
        """Duration range for CR pulses."""
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )


@dataclass
class HamiltonianTomographyCRAmplResults(Results):
    """HamiltonianTomographyCRAmpl outputs."""

    echo: bool
    hamiltonian_terms: dict[
        tuple[QubitId, QubitId], list[tuple[float, dict[HamiltonianTerm, float]]]
    ] = field(default_factory=dict)
    """Terms in effective Hamiltonian."""
    fitted_parameters: dict[tuple[QubitId, QubitId], dict[HamiltonianTerm, list]] = (
        field(default_factory=dict)
    )
    """Fitted parameters for Hamiltonian Terms values for different amplitudes."""

    control_pulse_amplitudes: dict[QubitPairId, dict[str, float]] = field(
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

    native: Literal["CNOT"] = "CNOT"
    """Two qubit interaction to be calibrated."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return all(key[:2] == pair for key in list(self.fitted_parameters))

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


@dataclass
class HamiltonianTomographyCRAmplData(HamiltonianTomographyData):
    """Data structure for CR Amplitude."""

    echo: bool
    amplitudes: list | None = None
    data: dict[
        tuple[QubitId, QubitId, Basis, SetControl],
        npt.NDArray[HamiltonianTomographyCRAmplType],
    ] = field(default_factory=dict)
    """Raw data acquired."""
    verbose_plot: bool = False
    """If `True` in the report all the single Hamiltonian tomographies are plotted."""

    @property
    def pairs(self):
        return {(i[0], i[1]) for i in self.data}

    def select_amplitude(self, amplitude: float):
        new_data = HamiltonianTomographyCRAmplData(
            echo=self.echo,
        )
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
    params: HamiltonianTomographyCRAmplParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRAmplData:
    """Data acquisition for Hamiltonian tomography CR protocol.

    We measure the expectation values X,Y and Z on the target qubit after
    applying the CR sequence specified by the input parameters. We repeat the
    measurement twice for each target qubit, once with the control qubit in state 0
    and once with the control qubit in state 1.

    We store the probability of the control qubit and the expectation value of the target qubit.

    """

    data = HamiltonianTomographyCRAmplData(
        echo=params.echo,
        amplitudes=params.amplitude_range.astype(float).tolist(),
        verbose_plot=params.verbose_plot,
    )

    for pair in targets:
        control, target = pair

        for basis in Basis:
            for setup in SetControl:
                sequence, cr_pulses, cr_target_pulses, delays = cr_sequence(
                    platform=platform,
                    control=control,
                    target=target,
                    duration=params.pulse_duration_end,
                    amplitude=params.control_ampl_end,
                    phase=params.control_phase,
                    target_amplitude=params.target_amplitude,
                    target_phase=params.target_phase,
                    echo=params.echo,
                    setup=setup,
                    basis=basis,
                )

                if params.interpolated_sweeper:
                    length_sweeper = Sweeper(
                        parameter=Parameter.duration_interpolated,
                        values=params.duration_range,
                        pulses=cr_pulses + cr_target_pulses,
                    )
                else:
                    length_sweeper = Sweeper(
                        parameter=Parameter.duration,
                        values=params.duration_range,
                        pulses=cr_pulses + cr_target_pulses + delays,
                    )

                amp_sweeper = Sweeper(
                    parameter=Parameter.amplitude,
                    values=params.amplitude_range,
                    pulses=cr_pulses,
                )

                updates = []
                updates.append(
                    {
                        platform.qubits[control].drive_extra[target]: {
                            "frequency": platform.config(
                                platform.qubits[target].drive
                            ).frequency
                        }
                    }
                )

                results = platform.execute(
                    [sequence],
                    [[length_sweeper], [amp_sweeper]],
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                    updates=updates,
                )
                target_acq_handle = list(
                    sequence.channel(platform.qubits[target].acquisition)
                )[-1].id
                control_acq_handle = list(
                    sequence.channel(platform.qubits[control].acquisition)
                )[-1].id

                prob_target = results[target_acq_handle].ravel()
                prob_control = results[control_acq_handle].ravel()

                data.register_qubit(
                    HamiltonianTomographyCRAmplType,
                    (control, target, basis, setup),
                    dict(
                        x=length_sweeper.values,
                        amp=amp_sweeper.values,
                        prob_target=1 - 2 * prob_target,
                        error_target=2
                        * np.sqrt(
                            EPS + prob_target * (1 - prob_target) / params.nshots
                        ),
                        prob_control=prob_control,
                        error_control=np.sqrt(
                            EPS + prob_control * (1 - prob_control) / params.nshots
                        ),
                    ),
                )

    return data


def _fit(
    data: HamiltonianTomographyCRAmplData,
) -> HamiltonianTomographyCRAmplResults:
    """Post-processing function for HamiltonianTomographyCRAmpl.

    We fit the expectation values using the Eq. S10 from the paper https://arxiv.org/pdf/2303.01427.
    Afterwards, we extract the Hamiltonian terms from the fitted parameters.

    """
    hamiltonian_terms, fitted_parameters, cal_amplitudes, ham_tom_params, cr_lengths = (
        amplitude_tomography_cr_fit(
            data=data,
        )
    )

    return HamiltonianTomographyCRAmplResults(
        echo=data.echo,
        hamiltonian_terms=hamiltonian_terms,
        fitted_parameters=fitted_parameters,
        control_pulse_amplitudes=cal_amplitudes,
        hamiltonian_tom_params=ham_tom_params,
        cr_lengths=cr_lengths,
        verbose_plot=data.verbose_plot,
    )


def _plot(
    data: HamiltonianTomographyCRAmplData,
    target: QubitPairId,
    fit: HamiltonianTomographyCRAmplResults,
):
    """Plotting function for HamiltonianTomographyCRAmpl."""
    figs, fitting_report = calibration_cr_plot(data, target, fit)

    if fit.verbose_plot:
        from qibocal.protocols.two_qubit_interaction.cross_resonance.hamiltonian_tomography.length import (
            HamiltonianTomographyCRLengthResults,
        )
        from qibocal.protocols.two_qubit_interaction.cross_resonance.hamiltonian_tomography.utils import (
            tomography_cr_plot,
        )

        for a in data.amplitudes:
            selected_ham_terms = fit.select_pair_and_ampl_ham_params(a, target)
            ampl_data = data.select_amplitude(a)
            ham_tom_fit = HamiltonianTomographyCRLengthResults(
                echo=fit.echo,
                hamiltonian_terms=selected_ham_terms,
                fitted_parameters=fit.hamiltonian_tom_params[a],
                cr_lengths=fit.cr_lengths[a],
            )
            f, _ = tomography_cr_plot(ampl_data, target, ham_tom_fit)
            figs += f
            if ham_tom_fit is not None:
                fitting_report += "\n" + table_html(
                    table_dict(
                        8 * [target],
                        [f"{term.name} [MHz]" for term in HamiltonianTerm]
                        + ["CR duration (ns)", "Cancellation amplitude [a.u.]"],
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

    return figs, fitting_report


hamiltonian_tomography_cr_amplitude = Routine(
    _acquisition, _fit, _plot, two_qubit_gates=True
)
"""HamiltonianTomographyCRAmplitude Routine object."""
