"""Hamiltonian tomography protocol for CR gate calibration.

This protocol computes the expectation values for X, Y and Z for the target qubit
after the application of a cross resonance sequence. The CR pulses are played on the control drive
channel with frequency set to the frequency of the target drive channel.
"""

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Sweeper,
)

from .....auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from .....calibration import CalibrationPlatform
from ..utils import Basis, SetControl, cr_sequence
from .length import HamiltonianTomographyCRLengthResults
from .utils import (
    EPS,
    HamiltonianTerm,
    amplitude_tomography_cr_fit,
    calibration_cr_plot,
    cyclic_prob,
)

HamiltonianTomographyCRAmplitudeType = np.dtype(
    [
        ("prob_target", np.float64),
        ("error_target", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
        ("amp", np.float64),
        ("x", np.int64),
    ]
)
"""Custom dtype for CR amplitude."""


@dataclass
class HamiltonianTomographyCRAmplitudeParameters(Parameters):
    """HamiltonianTomographyCRAmplitude runcard inputs."""

    target_calibration: bool
    """Sweep over control or target qubit amplitude"""
    pulse_duration_start: float
    """Initial duration of CR pulse [ns]."""
    pulse_duration_end: float
    """Final duration of CR pulse [ns]."""
    pulse_duration_step: float
    """Step CR pulse duration [ns]."""
    control_amplitude: float
    """Initial amplitude of CR pulse."""
    amplitude_end: float
    """Final amplitude of CR pulse."""
    amplitude_step: float
    """Step CR pulse amplitude."""
    target_amplitude: float
    """Amplitude of cancellation pulse."""
    control_phase: Union[dict[QubitPairId, float], float] = 0
    """Phase of CR pulse."""
    target_phase: Union[dict[QubitPairId, float], float] = 0
    """Phase of target pulse."""
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""
    echo: bool = False
    """Apply echo sequence or not.

    The ECR is described in https://arxiv.org/pdf/1210.7011
    """
    amplitude_plot_dict: dict[QubitPairId, list] = field(default_factory=dict)
    """
    Dictionary containing the values of amplitude for which plot hamiltonian tomography
    for each qubit.
    """

    @property
    def amplitude_range(self) -> np.ndarray:
        """Amplitude range for CR pulses."""
        return np.arange(
            self.control_amplitude
            if not self.target_calibration
            else self.target_amplitude,
            self.amplitude_end,
            self.amplitude_step,
        )

    @property
    def duration_range(self) -> np.ndarray:
        """Duration range for CR pulses."""
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )


@dataclass
class HamiltonianTomographyCRAmplitudeResults(Results):
    """HamiltonianTomographyCRAmplitude outputs."""

    target_calibration: bool
    """Sweep over control or target qubit amplitude"""

    hamiltonian_terms: dict[QubitPairId, dict] = field(default_factory=dict)
    """Terms in effective Hamiltonian."""
    fitted_parameters: dict[tuple[QubitId, QubitId], dict[HamiltonianTerm, list]] = (
        field(default_factory=dict)
    )
    """Fitted parameters for Hamiltonian Terms values for different amplitudes."""

    tomography_length_parameters: HamiltonianTomographyCRLengthResults = field(
        default_factory=dict
    )
    """Fitted parameters from X,Y,Z expectation values."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class HamiltonianTomographyCRAmplitudeData(Data):
    """Data structure for CR Amplitude."""

    target_calibration: bool
    amplitudes: list = None
    data: dict[
        tuple[QubitId, QubitId, Basis, SetControl],
        npt.NDArray[HamiltonianTomographyCRAmplitudeType],
    ] = field(default_factory=dict)
    """Raw data acquired."""
    amplitude_plot_dict: dict[QubitPairId, list] = field(default_factory=dict)
    """
    Dictionary containing the values of amplitude for which plot hamiltonian tomography
    for each qubit.
    """

    @property
    def pairs(self):
        return {(i[0], i[1]) for i in self.data}

    def select_amplitude(self, amplitude: float):
        new_data = HamiltonianTomographyCRAmplitudeData(
            target_calibration=self.target_calibration
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
        ar["error_target"] = data_dict["error_target"]
        ar["prob_control"] = data_dict["prob_control"].ravel()
        ar["error_control"] = data_dict["error_control"]

        self.data[data_keys] = np.rec.array(ar)


def _acquisition(
    params: HamiltonianTomographyCRAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRAmplitudeData:
    """Data acquisition for Hamiltonian tomography CR protocol.

    We measure the expectation values X,Y and Z on the target qubit after
    applying the CR sequence specified by the input parameters. We repeat the
    measurement twice for each target qubit, once with the control qubit in state 0
    and once with the control qubit in state 1.

    We store the probability of the control qubit and the expectation value of the target qubit.

    """

    data = HamiltonianTomographyCRAmplitudeData(
        target_calibration=params.target_calibration,
        amplitudes=params.amplitude_range.astype(float).tolist(),
        amplitude_plot_dict=params.amplitude_plot_dict,
    )

    for pair in targets:
        control, target = pair
        pair = (control, target)

        if params.target_calibration:
            ctrl_amplitude = params.control_amplitude
            target_amplitude = params.amplitude_end
        else:
            ctrl_amplitude = params.amplitude_end
            target_amplitude = params.target_amplitude

        for basis in Basis:
            for setup in SetControl:
                sequence, cr_pulses, cr_target_pulses, delays = cr_sequence(
                    platform=platform,
                    control=control,
                    target=target,
                    setup=setup,
                    amplitude=ctrl_amplitude,
                    phase=(
                        params.control_phase[pair]
                        if isinstance(params.control_phase, dict)
                        else params.control_phase
                    ),
                    target_amplitude=target_amplitude,
                    target_phase=(
                        params.target_phase[pair]
                        if isinstance(params.target_phase, dict)
                        else params.target_phase
                    ),
                    duration=params.pulse_duration_end,
                    echo=params.echo,
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
                    pulses=(
                        cr_target_pulses if params.target_calibration else cr_pulses
                    ),
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
                acquisition_type = AcquisitionType.INTEGRATION
                results = platform.execute(
                    [sequence],
                    [[length_sweeper], [amp_sweeper]],
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=acquisition_type,
                    averaging_mode=AveragingMode.CYCLIC,
                    updates=updates,
                )
                target_acq_handle = list(
                    sequence.channel(platform.qubits[target].acquisition)
                )[-1].id
                control_acq_handle = list(
                    sequence.channel(platform.qubits[control].acquisition)
                )[-1].id

                if acquisition_type is AcquisitionType.INTEGRATION:
                    prob_target = results[target_acq_handle]
                    prob_control = results[control_acq_handle]
                else:
                    prob_target = cyclic_prob(
                        results[target_acq_handle], state=1
                    ).ravel()
                    prob_control = cyclic_prob(
                        results[control_acq_handle], state=1
                    ).ravel()

                data.register_qubit(
                    HamiltonianTomographyCRAmplitudeType,
                    (control, target, basis, setup),
                    dict(
                        x=length_sweeper.values,
                        amp=amp_sweeper.values,
                        prob_target=1 - 2 * prob_target,
                        error_target=(
                            EPS
                            + 2
                            * np.sqrt(prob_target * (1 - prob_target) / params.nshots)
                        ).tolist(),
                        prob_control=prob_control,
                        error_control=(
                            EPS
                            + np.sqrt(prob_control * (1 - prob_control) / params.nshots)
                        ).tolist(),
                    ),
                )
    return data


def _fit(
    data: HamiltonianTomographyCRAmplitudeData,
) -> HamiltonianTomographyCRAmplitudeResults:
    """Post-processing function for HamiltonianTomographyCRAmplitude.

    We fit the expectation values using the Eq. S10 from the paper https://arxiv.org/pdf/2303.01427.
    Afterwards, we extract the Hamiltonian terms from the fitted parameters.

    """
    length_tom_params, hamiltonian_terms, fitted_parameters = (
        amplitude_tomography_cr_fit(
            data=data,
        )
    )
    length_tom_params = HamiltonianTomographyCRLengthResults(
        fitted_parameters=length_tom_params
    )

    return HamiltonianTomographyCRAmplitudeResults(
        target_calibration=data.target_calibration,
        hamiltonian_terms=hamiltonian_terms,
        fitted_parameters=fitted_parameters,
        tomography_length_parameters=length_tom_params,
    )


def _plot(
    data: HamiltonianTomographyCRAmplitudeData,
    target: QubitPairId,
    fit: HamiltonianTomographyCRAmplitudeResults,
):
    """Plotting function for HamiltonianTomographyCRAmplitude."""
    figs, fitting_report = calibration_cr_plot(data, target, fit)
    return figs, fitting_report


hamiltonian_tomography_cr_amplitude = Routine(_acquisition, _fit, _plot)
"""HamiltonianTomography Routine object."""
