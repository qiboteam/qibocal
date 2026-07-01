from qibolab import Delay, PulseLike, PulseSequence
from qibolab._core.identifier import ChannelId

from qibocal.auto.operation import QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.update import replace

from .parent_classes import InputError


def check_correct_drive_lines_setup(
    targets: list[QubitId], input_drivelines: list[QubitId] | None
) -> list[QubitId]:
    """Validate the drive lines assigned to target qubits."""

    drive_lines = input_drivelines if input_drivelines is not None else targets
    if len(drive_lines) != len(targets):
        raise InputError(
            "Each qubit has to be assigned to a drive line; "
            "If inserted, drive_lines must have the same length of targets list."
        )
    return drive_lines


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
        qubit_freq = platform.parameters.configs[qd_channel].frequency
        update |= {cross_channel: {"frequency": qubit_freq}}
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
        single_q_seq |= PulseSequence([platform.natives.single_qubit[q].MZ()[0]])

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
