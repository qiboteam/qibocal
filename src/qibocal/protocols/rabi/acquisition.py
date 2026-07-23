from qibolab import Delay, IqChannel, PulseLike, PulseSequence
from qibolab._core.identifier import ChannelId

from qibocal.auto.operation import QubitId, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.update import replace


def define_qubits_and_drivelines(
    targets: list[QubitId] | list[QubitPairId],
) -> tuple[list[QubitId], list[QubitId]]:
    """Separate target qubits from the drive lines used to address them.

    A single qubit target is interpreted as being driven by its own line,
    while a ``(qubit, drive_line)`` pair specifies a potentially different
    drive line.
    """

    pairs = [(t if isinstance(t, tuple) else (t, t)) for t in targets]
    qubit_list, drive_lines = map(list, zip(*pairs))

    return qubit_list, drive_lines


def single_qubit_rabi_sequence(
    target: QubitId,
    drive_line: QubitId,
    platform: CalibrationPlatform,
    pulse_duration: float | None,
    pulse_ampl: float | None,
    rx90: bool,
) -> tuple[PulseSequence, PulseLike, ChannelId, dict]:
    """Generate a single qubit Rabi sequence given a specific qubit and the line we want to drive it."""

    single_q_sequence = PulseSequence()
    update = {}
    natives_pulses = platform.natives.single_qubit[target]

    qd_channel, qd_pulse = natives_pulses.RX90()[0] if rx90 else natives_pulses.RX()[0]
    if target != drive_line:
        # used when q is being driven with another line (cross rabi)
        cross_channel = platform.qubits[drive_line].drive
        cross_channel_obj = platform.channels[cross_channel]
        qubit_channel_obj = platform.channels[qd_channel]
        update |= {
            cross_channel: {
                "frequency": platform.parameters.configs[qd_channel].frequency
            }
        }
        if all(
            [isinstance(ch, IqChannel) for ch in [qubit_channel_obj, cross_channel_obj]]
        ):
            q_lo_params = platform.parameters.configs[qubit_channel_obj.lo]
            update |= {
                cross_channel_obj.lo: {
                    "frequency": q_lo_params.frequency,
                    "power": q_lo_params.power,
                }
            }

        qd_channel = cross_channel

    if pulse_ampl is not None:
        qd_pulse = replace(qd_pulse, amplitude=pulse_ampl)

    if pulse_duration is not None:
        qd_pulse = replace(qd_pulse, duration=pulse_duration)

    if rx90:
        single_q_sequence.append((qd_channel, qd_pulse))

    single_q_sequence.append((qd_channel, qd_pulse))

    return single_q_sequence, qd_pulse, qd_channel, update


def sequence_amplitude(
    targets: list[QubitId],
    drive_lines: list[QubitId],
    platform: CalibrationPlatform,
    pulse_duration: float | None,
    pulse_ampl: float | None,
    rx90: bool,
) -> tuple[PulseSequence, list[PulseLike], dict[QubitId, float], dict]:
    """Generate Rabi pulse sequences for amplitude sweeping on multiple qubits and generic drive lines scheme."""

    sequence = PulseSequence()
    qd_pulses: list[PulseLike] = []
    durations: dict[QubitId, float] = {}
    updates = {}
    for q, d in zip(targets, drive_lines):
        # creating Rabi sequence for a (qubit, drive_line) pair
        single_q_seq, single_q_pulse, _, single_q_update = single_qubit_rabi_sequence(
            target=q,
            drive_line=d,
            platform=platform,
            pulse_duration=pulse_duration,
            pulse_ampl=pulse_ampl,
            rx90=rx90,
        )
        qd_pulses.append(single_q_pulse)
        durations[q] = single_q_pulse.duration
        updates |= single_q_update

        # aligning readout pulses to single qubit sequence
        single_q_seq |= PulseSequence(platform.natives.single_qubit[q].MZ())

        # adding the single qubit sequence to the complete one
        sequence += single_q_seq

    return sequence, qd_pulses, durations, updates


def sequence_length(
    targets: list[QubitId],
    drive_lines: list[QubitId],
    platform: CalibrationPlatform,
    pulse_duration: float | None,
    pulse_ampl: float | None,
    rx90: bool,
    use_align: bool = False,
) -> tuple[PulseSequence, list[PulseLike], list[Delay], dict[QubitId, float], dict]:
    """Generate Rabi pulse sequences for duration sweeping on multiple qubits and generic drive lines scheme."""

    sequence = PulseSequence()
    amplitudes: dict[QubitId, float] = {}
    updates = {}
    qd_pulses: list[PulseLike] = []
    delays: list[Delay] = []
    for q, d in zip(targets, drive_lines):
        # creating Rabi sequence for a (qubit, drive_line) pair
        single_q_seq, single_q_pulse, single_q_channel, single_q_update = (
            single_qubit_rabi_sequence(
                target=q,
                drive_line=d,
                platform=platform,
                pulse_duration=pulse_duration,
                pulse_ampl=pulse_ampl,
                rx90=rx90,
            )
        )
        sequence += single_q_seq
        qd_pulses.append(single_q_pulse)
        amplitudes[q] = single_q_pulse.amplitude
        updates |= single_q_update

        # appending readout pulses
        ro_channel, ro_pulse = platform.natives.single_qubit[q].MZ()[0]
        if use_align:
            sequence.align([single_q_channel, ro_channel])
        else:
            delays.append(Delay(duration=single_q_pulse.duration))
            sequence.append((ro_channel, delays[-1]))
        sequence.append((ro_channel, ro_pulse))

    return sequence, qd_pulses, delays, amplitudes, updates
