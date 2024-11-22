from dataclasses import dataclass

from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.update import replace

from ... import update
from ...result import magnitude, phase
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
    params: RabiAmplitudeEFParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
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
    for q in targets:
        natives = platform.natives.single_qubit[q]
        qd_channel, qd_pulse = natives.RX()[0]
        qd12_channel, qd12_pulse = natives.RX12()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        if params.pulse_length is not None:
            qd12_pulse = replace(qd_pulse, duration=params.pulse_length)

        durations[q] = qd12_pulse.duration
        qd_pulses[q] = qd12_pulse
        ro_pulses[q] = ro_pulse

        sequence.append((qd_channel, qd_pulse))
        sequence.append((qd12_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((qd12_channel, qd12_pulse))
        sequence.append(
            (qd_channel, Delay(duration=qd_pulse.duration + qd12_pulse.duration))
        )
        sequence.append((qd_channel, qd_pulse))
        sequence.append(
            (ro_channel, Delay(duration=2 * qd_pulse.duration + qd12_pulse.duration))
        )
        sequence.append((ro_channel, ro_pulse))

    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.min_amp, params.max_amp, params.step_amp),
        pulses=[qd_pulses[qubit] for qubit in targets],
    )

    if params.rx90:
        raise ValueError("Use RX90 = False")

    data = RabiAmplitudeEFData(durations=durations, rx90=False)

    # sweep the parameter
    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for qubit in targets:
        result = results[ro_pulses[qubit].id]
        data.register_qubit(
            amplitude_signal.RabiAmpSignalType,
            (qubit),
            dict(
                amp=sweeper.values,
                signal=magnitude(result),
                phase=phase(result),
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


def _update(
    results: RabiAmplitudeEFResults, platform: CalibrationPlatform, target: QubitId
):
    """Update RX2 amplitude_signal"""
    update.drive_12_amplitude(results.amplitude[target], results.rx90, platform, target)
    update.drive_12_duration(results.length[target], results.rx90, platform, target)


rabi_amplitude_ef = Routine(_acquisition, amplitude_signal._fit, _plot, _update)
"""RabiAmplitudeEF Routine object."""
