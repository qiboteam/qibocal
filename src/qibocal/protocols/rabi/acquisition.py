from qibolab import Delay, PulseLike, PulseSequence

from qibocal.auto.operation import Parameters, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.update import replace


def sequence_amplitude(
    targets: list[QubitId],
    params: Parameters,
    platform: CalibrationPlatform,
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
    drive_lines: list[QubitId],
    platform: CalibrationPlatform,
    pulse_duration: float | None,
    pulse_ampl: float | None,
    rx90: bool,
    use_align: bool = False,
) -> tuple[PulseSequence, list[PulseLike], list[Delay], dict, dict]:
    """Return sequence for rabi length for a list of qubits."""

    sequence = PulseSequence()
    amplitudes = {}
    updates = {}
    qd_pulses: list[PulseLike] = []
    delays: list[Delay] = []
    for q, d in zip(targets, drive_lines):
        q_seq, q_pulse, q_delay, q_ampl, q_update = single_qubit_sequence_length(
            target=q,
            drive_line=d,
            platform=platform,
            pulse_duration=pulse_duration,
            pulse_ampl=pulse_ampl,
            rx90=rx90,
            use_align=use_align,
        )

        sequence += q_seq
        qd_pulses.append(q_pulse)
        delays.append(q_delay)
        amplitudes[q] = q_ampl
        updates |= q_update

    return sequence, qd_pulses, delays, amplitudes, updates


def single_qubit_sequence_length(
    target: QubitId,
    drive_line: QubitId,
    platform: CalibrationPlatform,
    pulse_duration: float | None,
    pulse_ampl: float | None,
    rx90: bool,
    use_align: bool = False,
) -> tuple[PulseSequence, PulseLike, Delay, float, dict]:
    """Return sequence for rabi length for a single qubit."""

    natives_dict = platform.natives.single_qubit

    # qubit channels and pulses
    qd_channel, qd_pulse = (
        natives_dict[target].RX90()[0] if rx90 else natives_dict[target].RX()[0]
    )
    ro_channel, ro_pulse = natives_dict[target].MZ()[0]

    updates = {}
    if target != drive_line:
        # used when q is being driven with another line (cross rabi)
        cross_channel, _ = (
            natives_dict[drive_line].RX90()[0]
            if rx90
            else natives_dict[drive_line].RX()[0]
        )
        qubit_freq = platform.parameters.configs[qd_channel].frequency
        updates |= {platform.qubits[drive_line].drive: {"frequency": qubit_freq}}
        qd_channel = cross_channel

    if pulse_ampl is not None:
        qd_pulse = replace(qd_pulse, amplitude=pulse_ampl)

    if pulse_duration is not None:
        qd_pulse = replace(qd_pulse, duration=pulse_duration)

    sequence = PulseSequence()

    if rx90:
        sequence.append((qd_channel, qd_pulse))

    sequence.append((qd_channel, qd_pulse))
    if use_align:
        sequence.align([qd_channel, ro_channel])
    else:
        delay = Delay(duration=qd_pulse.duration)
        sequence.append((ro_channel, delay))
    sequence.append((ro_channel, ro_pulse))

    return sequence, qd_pulse, delay, qd_pulse.amplitude, updates
