import numpy as np
from qibolab import Delay, Platform, Pulse, PulseSequence, Rectangular, VirtualZ

from qibocal.auto.operation import (
    QubitId,
    QubitPairId,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import angle_wrap
from qibocal.update import cnot_sequence, replace

from .cr_parent_classes import Basis, SetControl


def cross_resonance_pulses(
    platform: CalibrationPlatform, control: QubitId, target: QubitId
) -> tuple[Pulse | None, Pulse | None]:
    """Retrieve cross-resonance (CR) pulse parameters from platform calibration.
    This function extracts the CR pulse and its corresponding cancellation pulse
    from the platform's native gates configuration for a given control-target qubit pair.
    """

    # extracting the channels involved in the CR gate
    target_channel, _ = platform.parameters.native_gates.single_qubit[target].RX()[0]
    cr_channel = platform.qubits[control].drive_extra[target]

    # CNOT sequence present in the platform parameters.json
    cnot_cal_seq = platform.parameters.native_gates.two_qubit[(control, target)].CNOT

    cr_params = None
    canc_params = None
    if cnot_cal_seq is not None and len(cnot_cal_seq) != 0:
        for ch, p in cnot_cal_seq[2:]:
            if cr_params is None and ch == cr_channel and isinstance(p, Pulse):
                cr_params = p
            if canc_params is None and ch == target_channel and isinstance(p, Pulse):
                canc_params = p

            if cr_params is not None and canc_params is not None:
                # here we identified both the CR and cancellation pulse,
                # we can stop looking at the remaining pulses
                break

    return cr_params, canc_params


def ro_delay_range(
    cr_pulse_duration_range: tuple[float, float, float],
    echo: bool,
    cntl_setup: SetControl,
    control: QubitId,
    platform: Platform,
) -> tuple[float, float, float]:
    """Delay range for RO pulses.

    add the number of the pi-pulses if we are in echo mode
    or if we want to set the control to 1.
    num_cr_pulses is the number of cross resonance pulses:
    == 1 if there is no echo sequence
    == 2 if there is echo sequence
    num_pi_pulses is the number of pi-pulses:
    == 0 if there is no echo and control at 0
    == 1 if there is no echo and control at 1
    == 2 if there is echo and control at 0
    == 3 if there is echo and control at 1
    """

    # Calculate number of CR pulses: 1 if no echo, 2 if echo is enabled
    num_cr_pulses = 1 + int(echo)
    # Calculate number of pi-pulses: 2 per echo, plus 1 if control is set to X (SetControl.X)
    num_pi_pulses = 2 * int(echo) + int(cntl_setup == SetControl.X)

    # Get the duration of the single-qubit RX (pi/2) pulse for the control qubit
    pi_pulse_duration = platform.natives.single_qubit[control].RX()[0][1].duration

    # the total duration of the cross resonance sequence is given by:
    # num_cr_pulses * cr_duration + num_pi_pulses * pi_pulse_duration

    # Calculate minimum delay: start of CR range + all pi-pulses
    tot_delay_start = (
        num_cr_pulses * cr_pulse_duration_range[0] + num_pi_pulses * pi_pulse_duration
    )
    # Calculate maximum delay: end of CR range + all pi-pulses
    tot_delay_end = (
        num_cr_pulses * cr_pulse_duration_range[1] + num_pi_pulses * pi_pulse_duration
    )
    # Calculate step size: CR pulse range step multiplied by number of CR pulses
    tot_delay_step = num_cr_pulses * cr_pulse_duration_range[2]

    return (tot_delay_start, tot_delay_end, tot_delay_step)


def cross_res_sequence(
    platform: Platform,
    control: QubitId,
    target: QubitId,
    duration: float,
    control_amplitude: float,
    control_phase: float,
    target_amplitude: float | None,
    target_phase: float,
    echo: bool,
    interpolated_sweeper: bool = False,
) -> tuple[PulseSequence, list[Pulse], list[Pulse], list[Delay]]:
    """Creates sequence for CR experiment on ``control`` and ``target`` qubits.

    With ``setup`` it is possible to set the control qubit to 1 or keep it at 0.
    If ``echo`` is set to ``True`` a ECR gate will be played.
    With ``basis`` it is possible to set the measurement basis. If it is not provided
    the default is Z."""

    cr_control_pulses = []
    cr_target_pulses = []
    # delays introduced by the cross resonance sequence
    cr_delays = []

    sequence = PulseSequence()

    natives_control = platform.natives.single_qubit[control]
    natives_target = platform.natives.single_qubit[target]

    control_drive_channel, control_drive_pulse = natives_control.RX()[0]
    target_drive_channel, _ = natives_target.RX()[0]
    cr_channel = platform.qubits[control].drive_extra[target]

    cr_drive_pulse, target_drive_pulse = cross_resonance_pulses(
        platform, control, target
    )

    cr_drive_pulse = Pulse(
        duration=duration,
        amplitude=control_amplitude,
        relative_phase=angle_wrap(control_phase),
        envelope=Rectangular() if cr_drive_pulse is None else cr_drive_pulse.envelope,
    )
    cr_control_pulses.append(cr_drive_pulse)

    if target_amplitude is None:
        target_drive_pulse = Delay(duration=duration)
    else:
        target_drive_pulse = Pulse(
            duration=duration,
            amplitude=target_amplitude,
            relative_phase=angle_wrap(target_phase),
            envelope=Rectangular()
            if target_drive_pulse is None
            else target_drive_pulse.envelope,
        )
    cr_target_pulses.append(target_drive_pulse)

    # delays introduced by cross resonance pulses
    cr_delays.append(Delay(duration=cr_drive_pulse.duration))
    # first cross resonance pulse
    sequence.append((cr_channel, cr_drive_pulse))
    sequence.append((target_drive_channel, target_drive_pulse))
    if interpolated_sweeper:
        _ = sequence.align([control_drive_channel, cr_channel, target_drive_channel])
    else:
        sequence.append((control_drive_channel, cr_delays[-1]))

    if echo:
        # phase-flipped cross resonance pulses
        cr_drive_pulse_flipped = replace(
            cr_drive_pulse.new(),
            amplitude=-cr_drive_pulse.amplitude,
        )
        cr_control_pulses.append(cr_drive_pulse_flipped)

        if target_amplitude is None:
            target_pulse_flipped = target_drive_pulse.new()
        else:
            target_pulse_flipped = replace(
                target_drive_pulse.new(),
                amplitude=-target_drive_pulse.amplitude,
            )
        cr_target_pulses.append(target_pulse_flipped)

        # delay introduced by echo sequence
        echo_delay = Delay(duration=control_drive_pulse.duration)

        # first echo pulse
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((cr_channel, echo_delay))
        sequence.append((target_drive_channel, echo_delay))

        # delays introduced by cross resonance pulses
        cr_delays.append(Delay(duration=cr_drive_pulse.duration))
        # second cross resonance pulse with flipped phase
        sequence.append((cr_channel, cr_drive_pulse_flipped))
        sequence.append((target_drive_channel, target_pulse_flipped))
        if interpolated_sweeper:
            _ = sequence.align(
                [control_drive_channel, cr_channel, target_drive_channel]
            )
        else:
            sequence.append((control_drive_channel, cr_delays[-1]))

        # second echo pulse
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((cr_channel, echo_delay))
        sequence.append((target_drive_channel, echo_delay))

    return sequence, cr_control_pulses, cr_target_pulses, cr_delays


def appending_ro_sequence(
    platform: Platform,
    control: QubitId,
    target: QubitId,
    exp_sequence: PulseSequence,
    basis: Basis,
    interpolated_sweeper: bool,
) -> tuple[PulseSequence, list[Delay]]:
    """Append a readout pulse sequence for two qubits with a specified delay.

    This function constructs a pulse sequence that applies readout operations to both
    control and target qubits with an initial delay on each readout channel.
    """
    natives_control = platform.natives.single_qubit[control]
    natives_target = platform.natives.single_qubit[target]

    # platform acquisition channels and pulses for control and target qubits
    ro_channel_control, ro_pulse_control = natives_control.MZ()[0]
    ro_channel_target, ro_pulse_target = natives_target.MZ()[0]

    # switching measurement basis for target and control lines and align all the others
    target_drive_channel, _ = natives_target.RX()[0]
    control_drive_channel, _ = natives_control.RX()[0]
    # delay
    ro_delays = [Delay(duration=exp_sequence.duration) for _ in range(2)]

    # basis on which measuring the state
    if basis == Basis.X:
        tom_angle = np.pi / 2
    else:  # only relevant for Y
        tom_angle = 0

    target_rotation = natives_target.R(theta=np.pi / 2, phi=tom_angle)[0][1]
    control_rotation = natives_control.R(theta=np.pi / 2, phi=tom_angle)[0][1]

    if interpolated_sweeper:
        # if using interpolated_sweepers I need all the channels to be aligned
        cr_channel = platform.qubits[control].drive_extra[target]

        # align all the channels of the sequence
        _ = exp_sequence.align(
            [
                cr_channel,
                target_drive_channel,
                control_drive_channel,
                ro_channel_control,
                ro_channel_target,
            ]
        )
    else:
        # delay to align the readout pulses of control and target qubits
        # done in every case, even when measuring Z-basis (no additional pulse on target)
        # to have the same timing for all the measurements
        flip_duration = max(target_rotation.duration, control_rotation.duration)
        flip_delay = Delay(
            duration=flip_duration,
        )
        exp_sequence.append((ro_channel_control, flip_delay))
        exp_sequence.append((ro_channel_target, flip_delay.new()))

        # wait the whole cr-sequence duration before starting the readout pulses
        exp_sequence.append((ro_channel_target, ro_delays[0]))
        exp_sequence.append((ro_channel_control, ro_delays[1]))

    # applying the tomography rotations on X and Y basis
    if basis != Basis.Z:
        exp_sequence += PulseSequence(
            [
                (target_drive_channel, target_rotation),
                (control_drive_channel, control_rotation),
            ]
        )

    # adding readout pulses
    exp_sequence.append((ro_channel_target, ro_pulse_target))
    exp_sequence.append((ro_channel_control, ro_pulse_control))

    return exp_sequence, ro_delays


def cross_resonance_experiment(
    platform: Platform,
    pair_list: list[QubitPairId],
    duration: float | dict[QubitPairId, float],
    ctrl_ampl: float | dict[QubitPairId, float],
    ctrl_phase: float | dict[QubitPairId, float],
    targ_ampl: float | dict[QubitPairId, float | None] | None,
    targ_phase: float | dict[QubitPairId, float],
    basis: Basis,
    setup: SetControl,
    echo: bool = False,
    interpolated_sweeper: bool = False,
) -> tuple[
    PulseSequence,
    dict[QubitPairId, Pulse],
    dict[QubitPairId, Pulse],
    dict[QubitPairId, Delay],
    dict[QubitPairId, Delay],
]:
    """Build the pulse sequence for a cross-resonance experiment.

    The sequence is created for one or more control-target qubit pairs. The
    control qubit can be prepared in either |0> or |1> depending on ``setup``.
    If ``echo`` is True, an echoed cross-resonance sequence is generated.
    The target measurement basis is selected with ``basis``.
    """

    parallel_cr_sequences = PulseSequence()
    parallel_cr_pulses: dict[QubitPairId, Pulse] = {}
    parallel_cr_target_pulses: dict[QubitPairId, Pulse] = {}
    parallel_cr_delays: dict[QubitPairId, Delay] = {}
    parallel_ro_delays: dict[QubitPairId, Delay] = {}
    for pair in pair_list:
        control, target = pair

        # adding pi-pulse if we want to set control to 1
        cntl_setup_sequence = PulseSequence()
        if setup == SetControl.X:
            control_drive_channel, control_drive_pulse = platform.natives.single_qubit[
                control
            ].RX()[0]
            cntl_setup_sequence.append((control_drive_channel, control_drive_pulse))

        cr_sequence, cr_pulses, cr_target_pulses, cr_delays = cross_res_sequence(
            platform=platform,
            control=control,
            target=target,
            duration=duration[pair] if isinstance(duration, dict) else duration,
            control_amplitude=ctrl_ampl[pair]
            if isinstance(ctrl_ampl, dict)
            else ctrl_ampl,
            control_phase=ctrl_phase[pair]
            if isinstance(ctrl_phase, dict)
            else ctrl_phase,
            target_amplitude=targ_ampl[pair]
            if isinstance(targ_ampl, dict)
            else targ_ampl,
            target_phase=targ_phase[pair]
            if isinstance(targ_phase, dict)
            else targ_phase,
            echo=echo,
            interpolated_sweeper=interpolated_sweeper,
        )

        total_sequence, ro_delays = appending_ro_sequence(
            platform=platform,
            control=control,
            target=target,
            exp_sequence=cr_sequence,
            basis=basis,
            interpolated_sweeper=interpolated_sweeper,
        )
        # aligning with the state preparation pulses
        total_sequence = cntl_setup_sequence | total_sequence

        parallel_cr_sequences += total_sequence
        parallel_cr_pulses |= {pair: cr_pulses}
        parallel_cr_target_pulses |= {pair: cr_target_pulses}
        parallel_cr_delays |= {pair: cr_delays}
        parallel_ro_delays |= {pair: ro_delays}

    return (
        parallel_cr_sequences,
        parallel_cr_pulses,
        parallel_cr_target_pulses,
        parallel_cr_delays,
        parallel_ro_delays,
    )


def update_cnot_from_fit(
    platform: CalibrationPlatform,
    pair: QubitPairId,
    cr_duration: float,
    cr_ampl: float,
    control_phase: float,
    canc_ampl: float | None,
    canc_phase: float,
    echo_flag: bool,
) -> None:
    """Update CNOT gate calibration from cross-resonance fit parameters.

    Constructs and updates the CNOT gate using cross-resonance pulses with
    fitted parameters, including single-qubit rotations and virtual Z phases.
    """
    ctrl, targ = pair

    cr_seq, _, _, _ = cross_res_sequence(
        platform=platform,
        control=ctrl,
        target=targ,
        duration=cr_duration,
        control_amplitude=cr_ampl,
        control_phase=control_phase,
        target_amplitude=canc_ampl,
        target_phase=canc_phase,
        echo=echo_flag,
    )

    target_single_qubit_operation = (
        platform.qubits[targ].drive,
        platform.natives.single_qubit[targ].R(theta=np.pi / 2, phi=0)[0][1],
    )

    control_single_qubit_operation = (
        platform.qubits[ctrl].drive,
        VirtualZ(phase=np.pi / 2),
    )

    new_cr_seq = (
        PulseSequence([control_single_qubit_operation, target_single_qubit_operation])
        | cr_seq
    )

    cnot_sequence(new_cr_seq, platform, pair)
