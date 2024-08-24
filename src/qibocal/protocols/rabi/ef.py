from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Routine

from . import amplitude_signal, utils


@dataclass
class RabiAmplitudeEFParameters(amplitude_signal.RabiAmplitudeSignalParameters):
    """RabiAmplitudeEF runcard inputs."""


@dataclass
class RabiAmplitudeEFResults(amplitude_signal.RabiAmplitudeSignalResults):
    """RabiAmplitudeEF outputs."""


@dataclass
class RabiAmplitudeEFData(amplitude_signal.RabiAmplitudeSignalData):
    """RabiAmplitude data acquisition."""


def _acquisition(
    params: RabiAmplitudeEFParameters, platform: Platform, targets: list[QubitId]
) -> RabiAmplitudeEFData:
    r"""
    Data acquisition for Rabi EF experiment sweeping amplitude.

    The rabi protocol is performed after exciting the qubit to state 1.
    This protocol allows to compute the amplitude of the RX12 pulse to excite
    the qubit to state 2 starting from state 1.

    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    rx_pulses = {}
    durations = {}
    for qubit in targets:
        rx_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        qd_pulses[qubit] = platform.create_RX_pulse(
            qubit, start=rx_pulses[qubit].finish
        )
        if params.pulse_length is not None:
            qd_pulses[qubit].duration = params.pulse_length

        durations[qubit] = qd_pulses[qubit].duration
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(rx_pulses[qubit])
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
        [qd_pulses[qubit] for qubit in targets],
        type=SweeperType.FACTOR,
    )

    data = RabiAmplitudeEFData(durations=durations)

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
    for qubit in targets:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            amplitude_signal.RabiAmpSignalType,
            (qubit),
            dict(
                amp=qd_pulses[qubit].amplitude * qd_pulse_amplitude_range,
                signal=result.magnitude,
                phase=result.phase,
            ),
        )
    return data


def _plot(
    data: RabiAmplitudeEFData, target: QubitId, fit: RabiAmplitudeEFResults = None
):
    """Plotting function for RabiAmplitude."""
    figures, report = utils.plot(data, target, fit)
    if report is not None:
        report = report.replace("Pi pulse", "Pi pulse 12")
    return figures, report


def _update(results: RabiAmplitudeEFResults, platform: Platform, target: QubitId):
    """Update RX2 amplitude_signal"""
    update.drive_12_amplitude(results.amplitude[target], platform, target)
    update.drive_12_duration(results.length[target], platform, target)


rabi_amplitude_ef = Routine(_acquisition, amplitude_signal._fit, _plot, _update)
"""RabiAmplitudeEF Routine object."""
