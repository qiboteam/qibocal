"""Protocol to calibrate target cancellation pulse for improving CR gate."""

from dataclasses import dataclass, field

import numpy as np
from hamiltonian_tomography.fitting import sin_fit
from scipy.optimize import root

from ....auto.operation import (
    Data,
    Parameters,
    QubitPairId,
    Results,
    Routine,
)
from ....calibration import CalibrationPlatform
from .hamiltonian_tomography.amplitude import (
    HamiltonianTomographyCRAmplitudeParameters,
    HamiltonianTomographyCRAmplitudeResults,
)
from .hamiltonian_tomography.amplitude import (
    _acquisition as amplitude_tomography_acquisition,
)
from .hamiltonian_tomography.amplitude import (
    _fit as amplitude_tomography_fit,
)
from .hamiltonian_tomography.length import (
    HamiltonianTomographyCRLengthParameters,
    HamiltonianTomographyCRLengthResults,
)
from .hamiltonian_tomography.length import (
    _acquisition as length_tomography_acquisition,
)
from .hamiltonian_tomography.length import (
    _fit as length_tomography_fit,
)
from .hamiltonian_tomography.phase import (
    HamiltonianTomographyCRPhaseParameters,
    HamiltonianTomographyCRPhaseResults,
)
from .hamiltonian_tomography.phase import (
    _acquisition as phase_tomography_acquisition,
)
from .hamiltonian_tomography.phase import (
    _fit as phase_tomography_fit,
)
from .hamiltonian_tomography.utils import HamiltonianTerm
from .utils import SetControl

TOL = 1e-3
TOL_PERIOD = 1  # ns
TOL_AMP = 10  # a.u.


def cancellation_find_gate_length(vals_id, vals_x):
    omega_id = np.sqrt(np.sum(vals_id**2, axis=0))
    period_id = 2 * np.pi / omega_id
    omega_x = np.sqrt(np.sum(vals_x**2, axis=0))
    period_x = 2 * np.pi / omega_x

    if abs(period_id - period_x / 2) <= TOL_PERIOD:
        return np.mean([period_id, period_x / 2]).astype(float)

    return


def cancellation_find_phases(vals_iy, vals_yz):
    def zy_func(x):
        return sin_fit(x, *vals_yz)

    phi0 = root(zy_func, x0=[0]).x

    def iy_func(x):
        return sin_fit(x, *vals_iy)

    phi1 = root(iy_func, x0=[0]).x

    return phi0, phi1


@dataclass
class CancellationPulseParameters(Parameters):
    """CrossResonanceLength runcard inputs."""

    # length parameters
    pulse_duration_start: float
    """Initial duration of CR pulse [ns]."""
    pulse_duration_end: float
    """Final duration of CR pulse [ns]."""
    pulse_duration_step: float
    """Step CR pulse duration [ns]."""

    # phase parameters
    ctrl_phase: float
    """Initial CR control pulse phase."""
    target_phase: float
    """Initial float cancellation pulse phase."""
    phase_end: float
    """Final CR pulse phase."""
    phase_step: float
    """CR pulse phase step."""

    # amplitude parameters
    control_ampl: float
    """Initial control CR pulse amplitude."""
    target_ampl: float
    """Initial target cancellation pulse amplitude."""
    target_ampl_end: float
    """cancellation pulse amplitude."""
    target_ampl_step: float
    """pulse amplitude step."""

    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""
    echo: bool = False
    """Apply echo sequence or not."""


@dataclass
class CancellationPulseResults(Results):
    """CancellationPulse outputs."""

    cr_length: dict[QubitPairId, float] = field(default_factory=dict)
    cr_phase: dict[QubitPairId, float] = field(default_factory=dict)
    control_amplitude: dict[QubitPairId, float] = field(default_factory=dict)
    cancellation_phase: dict[QubitPairId, float] = field(default_factory=dict)
    cancellation_amplitude: dict[QubitPairId, float] = field(default_factory=dict)

    def __contains__(self, pair: QubitPairId):
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class CancellationPulseData(Data):
    """Data structure for CR length."""

    contol_amplitude: float
    length_results: HamiltonianTomographyCRLengthResults = field(default_factory=dict)
    phase_results: HamiltonianTomographyCRPhaseResults = field(default_factory=dict)
    amplitude_results: HamiltonianTomographyCRAmplitudeResults = field(
        default_factory=dict
    )
    """Raw data acquired."""

    @property
    def pairs(self):
        return {(i[0], i[1]) for i in self.data}


def _acquisition(
    params: CancellationPulseParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CancellationPulseData:
    """Data acquisition for cross resonance protocol."""

    data = CancellationPulseData(contol_amplitude=params.control_ampl)

    length_tomography_params = HamiltonianTomographyCRLengthParameters(
        pulse_duration_start=params.pulse_duration_start,
        pulse_duration_end=params.pulse_duration_end,
        pulse_duration_step=params.pulse_duration_step,
        pulse_amplitude=params.control_ampl,
        interpolated_sweeper=params.interpolated_sweeper,
        echo=params.echo,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        hardware_average=params.hardware_average,
        classify=params.classify,
    )
    length_tomography_data = length_tomography_acquisition(
        length_tomography_params, platform, targets
    )
    length_tomography_res = length_tomography_fit(length_tomography_data)

    drive_phases = {}
    phase_tomography_params = HamiltonianTomographyCRPhaseParameters(
        target_calibration=False,
        pulse_duration_start=params.pulse_duration_start,
        pulse_duration_end=params.pulse_duration_end,
        pulse_duration_step=params.pulse_duration_step,
        control_amplitude=params.control_ampl,
        control_phase=params.ctrl_phase,
        target_amplitude=params.target_ampl,
        target_phase=params.target_phase,
        interpolated_sweeper=params.interpolated_sweeper,
        echo=params.echo,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        hardware_average=params.hardware_average,
        classify=params.classify,
    )

    phase_tomography_data = phase_tomography_acquisition(
        phase_tomography_params, platform, targets
    )
    phase_tomography_res = phase_tomography_fit(phase_tomography_data)

    for pair in targets:
        control, target = pair
        pair = (control, target)
        phi0, phi1 = cancellation_find_phases(
            data.phase_results.fitted_parameters[control, target][HamiltonianTerm.IY],
            data.phase_results.fitted_parameters[control, target][HamiltonianTerm.ZY],
        )
        drive_phases["control"][pair] = phi0
        drive_phases["target"][pair] = phi0 - phi1

    amplitude_tomography_params = HamiltonianTomographyCRAmplitudeParameters(
        target_calibration=False,
        pulse_duration_start=params.pulse_duration_start,
        pulse_duration_end=params.pulse_duration_end,
        pulse_duration_step=params.pulse_duration_step,
        control_amplitude=params.control_ampl,
        amplitude_end=params.target_ampl_end,
        amplitude_step=params.target_ampl_step,
        control_phase=drive_phases["control"],
        target_amplitude=params.target_ampl,
        target_phase=drive_phases["target"],
        interpolated_sweeper=params.interpolated_sweeper,
        echo=params.echo,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        hardware_average=params.hardware_average,
        classify=params.classify,
    )
    amplitude_tomography_data = amplitude_tomography_acquisition(
        amplitude_tomography_params, platform, targets
    )
    amplitude_tomography_res = amplitude_tomography_fit(amplitude_tomography_data)

    data.length_results = length_tomography_res
    data.phase_results = phase_tomography_res
    data.amplitude_results = amplitude_tomography_res

    return data


def _fit(
    data: CancellationPulseData,
) -> CancellationPulseResults:
    """Post-processing function for CrossResonanceLength.

    After fitting the data with dumped cosine function, the effective coupling
    is computed as specified in https://arxiv.org/pdf/1905.11480.

    """

    calibrated_gates = {}
    target_list = data.amplitude_results.hamiltonian_terms.keys()

    cr_lengths = {}
    drive_phases = {}
    for pair in target_list:
        control, target = pair
        pair = (control, target)

        cr_lengths[pair] = cancellation_find_gate_length(
            data.length_results.fitted_parameters[control, target, SetControl.Id],
            data.length_results.fitted_parameters[control, target, SetControl.X],
        )

        phi0, phi1 = cancellation_find_phases(
            data.phase_results.fitted_parameters[control, target][HamiltonianTerm.IY],
            data.phase_results.fitted_parameters[control, target][HamiltonianTerm.ZY],
        )
        drive_phases["control"][pair] = phi0
        drive_phases["target"][pair] = phi0 - phi1

        ix_a, ix_b = data.amplitude_results.fitted_parameters[pair][HamiltonianTerm.IX]
        ix_line_root = -ix_b / ix_a

        iy_a, iy_b = data.amplitude_results.fitted_parameters[pair][HamiltonianTerm.IY]
        iy_line_root = -iy_b / iy_a

        if abs(ix_line_root - iy_line_root) <= TOL_AMP:
            calibrated_gates["cr_length"][pair] = cr_lengths[pair]
            calibrated_gates["ctrl_ampl"][pair] = data.control_amplitude
            calibrated_gates["ctrl_phase"][pair] = drive_phases[pair]["control"].astype(
                float
            )
            calibrated_gates["target_ampl"][pair] = np.mean(
                [ix_line_root, iy_line_root]
            ).astype(float)
            calibrated_gates["target_phase"][pair] = drive_phases[pair][
                "target"
            ].astype(float)

    return CancellationPulseResults(
        cr_length=calibrated_gates["cr_length"],
        cr_phase=calibrated_gates["ctrl_phase"],
        control_amplitude=calibrated_gates["ctrl_ampl"],
        cancellation_phase=calibrated_gates["target_phase"],
        cancellation_amplitude=calibrated_gates["target_ampl"],
    )


def _update(
    fit: CancellationPulseResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    return


cross_resonance_length = Routine(_acquisition, _fit, _update)
"""CrossResonance Routine object."""
