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
class RabiLengthParameters(Parameters):
    """RabiLength runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration (ns)."""
    pulse_duration_end: float
    """Final pi pulse duration (ns)."""
    pulse_duration_step: float
    """Step pi pulse duration (ns)."""
    pulse_amplitude: Optional[float] = None
    """Pi pulse amplitude. Same for all qubits."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class RabiLengthResults(Results):
    """RabiLength outputs."""

    length: dict[QubitId, int] = field(metadata=dict(update="drive_length"))
    """Pi pulse duration for each qubit."""
    amplitude: dict[QubitId, float] = field(metadata=dict(update="drive_amplitude"))
    """Pi pulse amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


RabiLenType = np.dtype(
    [("length", np.float64), ("msr", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthData(Data):
    """RabiLength acquisition outputs."""

    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiLenType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, length, msr, phase):
        """Store output for single qubit."""
        # to be able to handle the non-sweeper case
        shape = (1,) if np.isscalar(length) else length.shape
        ar = np.empty(shape, dtype=RabiLenType)
        ar["length"] = length
        ar["msr"] = msr
        ar["phase"] = phase
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: RabiLengthParameters, platform: Platform, qubits: Qubits
) -> RabiLengthData:
    r"""
    Data acquisition for RabiLength Experiment.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    amplitudes = {}
    for qubit in qubits:
        # TODO: made duration optional for qd pulse?
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
        if params.pulse_amplitude is not None:
            qd_pulses[qubit].amplitude = params.pulse_amplitude
        amplitudes[qubit] = qd_pulses[qubit].amplitude

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )

    sweeper = Sweeper(
        Parameter.duration,
        qd_pulse_duration_range,
        [qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.ABSOLUTE,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse length
    data = RabiLengthData(amplitudes=amplitudes)

    # execute the sweep
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
            length=qd_pulse_duration_range,
            msr=result.magnitude,
            phase=result.phase,
        )
    return data


def _fit(data: RabiLengthData) -> RabiLengthResults:
    """Post-processing for RabiLength experiment."""

    qubits = data.qubits
    fitted_parameters = {}
    durations = {}

    for qubit in qubits:
        pi_pulse_parameter = 0
        fitted_parameters = [0] * 4

        durations[qubit] = pi_pulse_parameter
        fitted_parameters[qubit] = [0, 0, 0, 0, 0]

    return RabiLengthResults(durations, data.amplitudes, fitted_parameters)


def _plot(data: RabiLengthData, fit: RabiLengthResults, qubit):
    """Plotting function for RabiLength experiment."""
    return utils.plot(data, fit, qubit)


rabi_length = Routine(_acquisition, _fit, _plot)
"""RabiLength Routine object."""
