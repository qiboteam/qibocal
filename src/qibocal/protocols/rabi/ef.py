from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import readout_frequency
from qibocal.update import drive_12_amplitude, drive_12_duration, replace

from .acquisition import check_correct_drive_lines_setup
from .amplitude_signal import _fit
from .parent_classes import (
    RabiAmplitudeParameters,
    RabiData,
    RabiResults,
)
from .processing import plot_signal

__all__ = ["rabi_amplitude_ef"]


RabiEFSignalType = np.dtype([("amp", np.float64), ("i", np.float64), ("q", np.float64)])
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiEFSignalData(RabiData):
    """RabiAmplitudeSignal data acquisition."""

    data: dict[QubitId, npt.NDArray[RabiEFSignalType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiEFSignalData:
    r"""
    Data acquisition for Rabi EF experiment sweeping amplitude.

    The rabi protocol is performed after exciting the qubit to state 1.
    This protocol allows to compute the amplitude of the RX12 pulse to excite
    the qubit to state 2 starting from state 1.

    """

    drive_lines = check_correct_drive_lines_setup(
        targets=targets, input_drivelines=params.drive_lines
    )

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    durations = {}
    updates = {}
    for q, d in zip(targets, drive_lines):
        natives = platform.natives.single_qubit[q]
        qd_channel, qd_pulse = natives.RX()[0]
        qd12_channel, qd12_pulse = natives.RX12()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        if q != d:
            # used when q is being driven with another line (cross rabi)
            cross_channel = platform.qubits[d].drive
            qubit12_freq = platform.parameters.configs[qd12_channel].frequency
            updates |= {cross_channel: {"frequency": qubit12_freq}}
            qd12_channel = cross_channel

        if params.pulse_length is not None:
            qd12_pulse = replace(qd12_pulse, duration=params.pulse_length)

        durations[q] = qd12_pulse.duration
        qd_pulses[q] = qd12_pulse
        ro_pulses[q] = ro_pulse

        sequence.append((qd_channel, qd_pulse))
        sequence.append((qd12_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((qd12_channel, qd12_pulse))
        sequence.append(
            (qd_channel, Delay(duration=qd_pulse.duration + qd12_pulse.duration))
        )
        sequence.append(
            (ro_channel, Delay(duration=qd_pulse.duration + qd12_pulse.duration))
        )
        sequence.append((ro_channel, ro_pulse))

    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.min_amp, params.max_amp, params.step_amp),
        pulses=[qd_pulses[qubit] for qubit in targets],
    )

    assert not params.rx90, "Rabi ef available only for RX pulses."

    data = RabiEFSignalData(
        drive_lines={t: d for t, d in zip(targets, drive_lines)},
        durations=durations,
        rx90=False,
    )

    # for signal measurement we have to change readout
    updates |= {
        platform.qubits[q].probe: {"frequency": readout_frequency(q, platform, state=1)}
        for q in targets
    }

    # sweep the parameter
    results = platform.execute(
        [sequence],
        [[sweeper]],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for qubit in targets:
        result = results[ro_pulses[qubit].id]
        data.register_qubit(
            RabiEFSignalType,
            (qubit),
            dict(
                amp=sweeper.values,
                i=result[..., 0],
                q=result[..., 1],
            ),
        )
    return data


def _plot(data: RabiEFSignalData, target: QubitId, fit: RabiResults = None):
    """Plotting function for RabiAmplitude."""
    figures, report = plot_signal(data, target, fit, data.rx90)
    if report is not None:
        report = report.replace("Pi pulse", "Pi pulse 12")
    return figures, report


def _update(results: RabiResults, platform: CalibrationPlatform, target: QubitId):
    """Update RX2 amplitude_signal"""
    if target == results.drive_lines[target]:
        drive_12_amplitude(results.amplitude[target], platform, target)
        drive_12_duration(results.length[target], platform, target)


rabi_amplitude_ef = Protocol(_acquisition, _fit, _plot, _update)
"""RabiAmplitudeEF Protocol object."""
