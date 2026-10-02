from dataclasses import dataclass

from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.update import replace

from ...result import magnitude, phase
from ..utils import readout_frequency
from . import utils
from .length_signal import (
    RabiLengthSignalData,
    RabiLengthSignalParameters,
    RabiLengthSignalResults,
    RabiLenSignalType,
    _fit,
)

__all__ = ["rabi_length_ef"]


@dataclass
class RabiLengthEFParameters(RabiLengthSignalParameters):
    """RabiLengthEF runcard inputs."""


@dataclass
class RabiLengthEFResults(RabiLengthSignalResults):
    """RabiLengthEF outputs."""


@dataclass
class RabiLengthEFData(RabiLengthSignalData):
    """RabiLengthEF data acquisition."""


def _acquisition(
    params: RabiLengthEFParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiLengthEFData:
    r"""
    Data acquisition for Rabi EF experiment sweeping duration.

    The rabi protocol is performed after exciting the qubit to state 1.
    This protocol allows to compute the duration of the RX12 pulse to excite
    the qubit to state 2 starting from state 1.

    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    delays = {}
    ro_pulses = {}
    amplitudes = {}
    for q in targets:
        natives = platform.natives.single_qubit[q]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]
        qd12_channel = platform.qubits[q].drive_extra[1, 2]
        if natives.RX12 is not None:
            [(_, qd12_pulse)] = natives.RX12()
            if params.pulse_amplitude is not None:
                qd12_pulse = replace(qd12_pulse, amplitude=params.pulse_amplitude)
        else:
            assert params.pulse_amplitude is not None
            qd12_pulse = Pulse(
                amplitude=params.pulse_amplitude,
                duration=params.pulse_duration_start,
                envelope=Rectangular(),
            )

        amplitudes[q] = qd12_pulse.amplitude
        qd_pulses[q] = qd12_pulse
        ro_pulses[q] = ro_pulse

        sequence.append((qd_channel, qd_pulse))
        sequence.append((qd12_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((qd12_channel, qd12_pulse))
        if params.interpolated_sweeper:
            sequence.align([qd_channel, qd12_channel, ro_channel])
        else:
            # the readout has to wait for the (fixed) RX pulse and for the
            # RX12 pulse, whose duration is swept together with this delay
            delays[q] = Delay(duration=16)
            sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
            sequence.append((ro_channel, delays[q]))
        sequence.append((ro_channel, ro_pulse))

    sweep_range = (
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )
    if params.interpolated_sweeper:
        sweeper = Sweeper(
            parameter=Parameter.duration_interpolated,
            range=sweep_range,
            pulses=[qd_pulses[q] for q in targets],
        )
    else:
        sweeper = Sweeper(
            parameter=Parameter.duration,
            range=sweep_range,
            pulses=[qd_pulses[q] for q in targets] + [delays[q] for q in targets],
        )

    assert not params.rx90, "Rabi ef available only for RX pulses."

    data = RabiLengthEFData(amplitudes=amplitudes, rx90=False)

    # sweep the parameter
    results = platform.execute(
        [sequence],
        [[sweeper]],
        updates=[
            {
                platform.qubits[q].probe: {
                    "frequency": readout_frequency(q, platform, state=1)
                }
            }
            for q in targets
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for q in targets:
        result = results[ro_pulses[q].id]
        data.register_qubit(
            RabiLenSignalType,
            (q),
            {
                "length": sweeper.values,
                "signal": magnitude(result),
                "phase": phase(result),
            },
        )
    return data


def _plot(data: RabiLengthEFData, target: QubitId, fit: RabiLengthEFResults = None):
    """Plotting function for RabiLengthEF."""
    figures, report = utils.plot(data, target, fit, data.rx90)
    if report is not None:
        report = report.replace("Pi pulse", "Pi pulse 12")
    return figures, report


def _update(
    results: RabiLengthEFResults, platform: CalibrationPlatform, target: QubitId
):
    """Update RX12 duration"""
    if results.length[target] is None:
        return

    rx12 = platform.natives.single_qubit[target].RX12
    if rx12 is not None:
        rx12_seq = [
            (
                rx12[0][0],
                replace(
                    rx12[0][1],
                    amplitude=results.amplitude[target],
                    duration=results.length[target],
                ),
            )
        ]
    else:
        amplitude = results.amplitude[target]
        assert isinstance(amplitude, float)
        duration = results.length[target]
        assert isinstance(duration, float)
        rx12_seq = [
            (
                platform.qubits[target].drive_extra[1, 2],
                Pulse(
                    amplitude=amplitude,
                    duration=duration,
                    envelope=Rectangular(),
                ),
            )
        ]
    platform.update({f"native_gates.single_qubit.{target}.RX12": rx12_seq})


rabi_length_ef = Protocol(_acquisition, _fit, _plot, _update)
"""RabiLengthEF Protocol object."""
