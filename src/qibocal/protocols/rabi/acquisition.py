from qibolab import Delay, Platform, PulseSequence

from qibocal.auto.operation import Parameters, QubitId
from qibocal.update import replace


def sequence_amplitude(
    targets: list[QubitId],
    params: Parameters,
    platform: Platform,
    rx90: bool,
) -> tuple[PulseSequence, dict, dict, dict]:
    """Return sequence for rabi amplitude."""

    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    durations = {}
    for q in targets:
        natives = platform.natives.single_qubit[q]

        qd_channel, qd_pulse = natives.RX90()[0] if rx90 else natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        if params.pulse_length is not None:
            qd_pulse = replace(qd_pulse, duration=params.pulse_length)

        durations[q] = qd_pulse.duration
        qd_pulses[q] = qd_pulse
        ro_pulses[q] = ro_pulse

        if rx90:
            sequence.append((qd_channel, qd_pulses[q]))

        sequence.append((qd_channel, qd_pulses[q]))
        sequence.append((ro_channel, Delay(duration=durations[q])))
        sequence.append((ro_channel, ro_pulse))
    return sequence, qd_pulses, ro_pulses, durations


def sequence_length(
    targets: list[QubitId],
    params: Parameters,
    platform: Platform,
    rx90: bool,
    use_align: bool = False,
) -> tuple[PulseSequence, dict, dict, dict]:
    """Return sequence for rabi length."""

    sequence = PulseSequence()
    qd_pulses = {}
    delays = {}
    ro_pulses = {}
    amplitudes = {}
    for q in targets:
        natives = platform.natives.single_qubit[q]

        qd_channel, qd_pulse = natives.RX90()[0] if rx90 else natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        if params.pulse_amplitude is not None:
            qd_pulse = replace(qd_pulse, amplitude=params.pulse_amplitude)

        amplitudes[q] = qd_pulse.amplitude
        ro_pulses[q] = ro_pulse
        qd_pulses[q] = qd_pulse

        if rx90:
            sequence.append((qd_channel, qd_pulse))

        sequence.append((qd_channel, qd_pulse))
        if use_align:
            sequence.align([qd_channel, ro_channel])
        else:
            delays[q] = Delay(duration=16)
            sequence.append((ro_channel, delays[q]))
        sequence.append((ro_channel, ro_pulse))

    return sequence, qd_pulses, delays, ro_pulses, amplitudes
