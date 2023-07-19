from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from . import utils


@dataclass
class RabiAmplitudeParameters(Parameters):
    """RabiAmplitude runcard inputs."""

    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    pulse_length: Optional[float]
    """RX pulse duration (ns)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class RabiAmplitudeResults(Results):
    """RabiAmplitude outputs."""

    amplitude: dict[QubitId, float] = field(metadata=dict(update="drive_amplitude"))
    """Drive amplitude for each qubit."""
    length: dict[QubitId, float] = field(metadata=dict(update="drive_length"))
    """Drive pulse duration. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitted parameters."""


RabiAmpType = np.dtype(
    [("amp", np.float64), ("msr", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeData(Data):
    """RabiAmplitude data acquisition."""

    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiAmpType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, amp, msr, phase):
        """Store output for single qubit."""
        ar = np.empty(amp.shape, dtype=RabiAmpType)
        ar["amp"] = amp
        ar["msr"] = msr
        ar["phase"] = phase
        self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: RabiAmplitudeParameters, platform: Platform, qubits: Qubits
) -> RabiAmplitudeData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    durations = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        if params.pulse_length is not None:
            qd_pulses[qubit].duration = params.pulse_length

        durations[qubit] = qd_pulses[qubit].duration
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse amplitude
    qd_pulse_amplitude_range = np.arange(
        params.min_amp_factor,
        params.max_amp_factor,
        params.step_amp_factor,
    )
    sweeper = Sweeper(
        Parameter.amplitude,
        qd_pulse_amplitude_range,
        [qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.FACTOR,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse amplitude
    data = RabiAmplitudeData(durations=durations)

    # sweep the parameter
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit,
            amp=qd_pulses[qubit].amplitude * qd_pulse_amplitude_range,
            msr=result.magnitude,
            phase=result.phase,
        )
    return data


def _fit(data: RabiAmplitudeData) -> RabiAmplitudeResults:
    """Post-processing for RabiAmplitude."""
    qubits = data.qubits

    pi_pulse_amplitudes = {}
    fitted_parameters = {}

    for qubit in qubits:
        pi_pulse_parameter = 0
        fitted_parameters = [0] * 4

        pi_pulse_amplitudes[qubit] = pi_pulse_parameter
        fitted_parameters[qubit] = [0, 0, 0, 0]

    return RabiAmplitudeResults(pi_pulse_amplitudes, data.durations, fitted_parameters)


def _plot(data: RabiAmplitudeData, fit: RabiAmplitudeResults, qubit):
    """Plotting function for RabiAmplitude."""
    return utils.plot(data, fit, qubit)


rabi_amplitude = Routine(_acquisition, _fit, _plot)
"""RabiAmplitude Routine object."""
