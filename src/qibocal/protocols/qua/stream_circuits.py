import math
from itertools import product
from typing import Optional, Union

import numpy as np
from qibolab import Platform, Pulse, Qubit, VirtualZ
from qm import CompilerOptionArguments, generate_qua_script
from qm.qua import (
    advance_input_stream,
    align,
    assign,
    case_,
    declare,
    declare_input_stream,
    declare_stream,
    dual_demod,
    fixed,
    for_,
    frame_rotation_2pi,
    measure,
    play,
    program,
    reset_frame,
    save,
    stream_processing,
    strict_timing_,
    switch_,
    wait,
)

from qibocal.config import log

from .configuration import baked_duration, generate_config

Sequence = list[tuple[str, str]]
QubitId = Union[int, str]
NATIVE_GATES = ["i", "x180", "y180", "x90", "y90", "-x90", "-y90"]
NATIVE_GATES_PAIRS = list(product(NATIVE_GATES, NATIVE_GATES))
NATIVE_GATES_PAIRS.append(("cz", "cz"))


def _convert_identity(g: Optional[str]) -> str:
    return "i" if g is None else g


def to_indices(sequence: Sequence) -> list[int]:
    return [
        NATIVE_GATES_PAIRS.index((_convert_identity(g0), _convert_identity(g1)))
        for g0, g1 in sequence
    ]


def classify(platform: Platform, qubit: Qubit, signal_i, signal_q, state):
    acquisition_config = platform.config(qubit.acquisition)
    threshold = acquisition_config.threshold
    angle = acquisition_config.iq_angle
    assign(state, np.cos(angle) * signal_i - np.sin(angle) * signal_q > threshold)


def find_flux_duration(platform: Platform, qubit0: QubitId, qubit1: QubitId) -> int:
    cz = platform.natives.two_qubit[(qubit0, qubit1)].CZ
    pulse = [p for _, p in cz if isinstance(p, Pulse)][0]
    return baked_duration(pulse.duration)


def find_cz_phase_correction(platform: Platform, qubit0: QubitId, qubit1: QubitId):
    cz = platform.natives.two_qubit[(qubit0, qubit1)].CZ
    virtalzs = [p for p in cz if isinstance(p[1], VirtualZ)]
    assert len(virtalzs) == 2
    ch0, vz0 = virtalzs[0]
    ch1, vz1 = virtalzs[1]
    if ch0 == platform.qubits[qubit1].drive:
        assert ch1 == platform.qubits[qubit0].drive
        return vz1.phase, vz0.phase
    else:
        assert ch1 == platform.qubits[qubit1].drive
    return vz0.phase, vz1.phase


def find_drive_duration(platform: Platform, qubit: QubitId) -> int:
    native = platform.natives.single_qubit[qubit].RX
    return math.ceil(native[0][1].duration)


def find_measurement_duration(platform: Platform, qubit: QubitId) -> int:
    native = platform.natives.single_qubit[qubit].MZ
    return math.ceil(native[0][1].duration)


# qubit0 is always the one the flux pulse is applied
def generate_program(
    platform: Platform,
    targets: list[QubitId],
    ncircuits: int,
    nshots: int,
    relaxation_time: int,
    max_depth: int = 128,
    save_input_streams: bool = False,
):
    assert len(targets) == 2
    qubit0 = platform.qubits[targets[0]]
    qubit1 = platform.qubits[targets[1]]

    channels = [
        qubit0.flux,
        qubit0.drive,
        qubit1.drive,
        qubit0.acquisition,
        qubit1.acquisition,
    ]

    drive_duration = find_drive_duration(platform, targets[0])
    assert find_drive_duration(platform, targets[1]) == drive_duration
    phase0, phase1 = find_cz_phase_correction(platform, *targets)

    with program() as prog:
        gates = declare_input_stream(int, name="gates_input_stream", size=max_depth)
        depth = declare_input_stream(int, name="depth_input_stream")

        i0 = declare(fixed)
        q0 = declare(fixed)
        i1 = declare(fixed)
        q1 = declare(fixed)
        state0 = declare(bool)
        state1 = declare(bool)
        state0_st = declare_stream()
        state1_st = declare_stream()

        if save_input_streams:
            depth_st = declare_stream()
            gates_st = declare_stream()

        icircuit = declare(int)
        with for_(icircuit, 0, (icircuit < ncircuits), (icircuit + 1)):
            advance_input_stream(depth)
            advance_input_stream(gates)
            ishot = declare(int)

            if save_input_streams:
                save(depth, depth_st)
                isave = declare(int)
                with for_(isave, 0, isave < max_depth, isave + 1):
                    save(gates[isave], gates_st)

            with for_(ishot, 0, (ishot < nshots), (ishot + 1)):
                reset_frame(qubit0.drive)
                reset_frame(qubit1.drive)
                wait(relaxation_time // 4)
                align(*channels)
                with strict_timing_():
                    igate = declare(int)
                    with for_(igate, 0, igate < depth, igate + 1):
                        align(qubit0.flux, qubit0.drive, qubit1.drive)
                        with switch_(gates[igate], unsafe=True):
                            for ig, (op0, op1) in enumerate(NATIVE_GATES_PAIRS):
                                with case_(ig):
                                    if op0 == "cz":
                                        assert op1 == "cz"
                                        wait(4, qubit0.flux)
                                        play("cz", qubit0.flux)
                                        frame_rotation_2pi(
                                            phase0 / (2 * np.pi), qubit0.drive
                                        )
                                        frame_rotation_2pi(
                                            phase1 / (2 * np.pi), qubit1.drive
                                        )
                                        wait(4, qubit0.flux)
                                    else:
                                        if op0 != "i":
                                            play(op0, qubit0.drive)
                                        if op1 != "i":
                                            play(op1, qubit1.drive)

                align(*channels)
                measure(
                    "measure",
                    qubit0.acquisition,
                    dual_demod.full("cos", "sin", i0),
                    dual_demod.full("minus_sin", "cos", q0),
                )
                measure(
                    "measure",
                    qubit1.acquisition,
                    dual_demod.full("cos", "sin", i1),
                    dual_demod.full("minus_sin", "cos", q1),
                )
                classify(platform, qubit0, i0, q0, state0)
                classify(platform, qubit1, i1, q1, state1)
                save(state0, state0_st)
                save(state1, state1_st)

        with stream_processing():
            state0_st.buffer(nshots).buffer(ncircuits).save("state0")
            state1_st.buffer(nshots).buffer(ncircuits).save("state1")
            if save_input_streams:
                depth_st.buffer(ncircuits).save("depths")
                gates_st.buffer(max_depth).buffer(ncircuits).save("gates")

    return prog


def estimate_duration(
    circuits, rx_duration, mz_duration, nshots, relaxation_time
) -> int:
    duration = sum(len(circuit) * rx_duration for circuit in circuits)
    duration += len(circuits) * (mz_duration + relaxation_time)
    return duration * nshots / 1e9


def execute(
    sequences: list[Sequence],
    platform: Platform,
    targets: list[QubitId],
    nshots: int,
    relaxation_time: Optional[float] = None,
    debug: Optional[str] = None,
):
    circuits = [to_indices(circuit) for circuit in sequences]

    max_depth = max(len(circuit) for circuit in circuits)
    program = generate_program(
        platform,
        sorted(targets)[::-1],  # FIXME: This will only work for qw5q_platinum
        ncircuits=len(circuits),
        nshots=nshots,
        relaxation_time=relaxation_time,
        max_depth=max_depth,
    )

    estimated_duration = estimate_duration(
        circuits,
        find_drive_duration(platform, targets[0]),
        find_measurement_duration(platform, targets[0]),
        nshots,
        relaxation_time,
    )
    log.info("Estimated duration: %.5f sec" % estimated_duration)

    # FIXME: This will only work for qw5q_platinum
    config = generate_config(
        platform, list(platform.qubits.keys()), sorted(targets)[::-1]
    )

    qmm = platform._controller.manager

    if debug is not None:
        with open(debug, "w") as file:
            file.write(generate_qua_script(program, config))

    qm = qmm.open_qm(config)
    job = qm.execute(
        program, compiler_options=CompilerOptionArguments(flags=["not-strict-timing"])
    )

    # TODO: Progress bar
    for circuit in circuits:
        job.push_to_input_stream("depth_input_stream", len(circuit))
        job.push_to_input_stream("gates_input_stream", circuit)

    handles = job.result_handles
    handles.wait_for_all_values()

    state0 = handles.get("state0").fetch_all()
    state1 = handles.get("state1").fetch_all()

    return state0, state1
