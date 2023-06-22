import numpy as np
from qibolab import AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Qubits, Routine

from .allxy import AllXYData, AllXYParameters, _fit, _plot

# from qibocal.data import Data


gatelist = [
    ["I", "I"],
    ["RX(pi)", "RX(pi)"],
    ["RY(pi)", "RY(pi)"],
    ["RX(pi)", "RY(pi)"],
    ["RY(pi)", "RX(pi)"],
    ["RX(pi/2)", "I"],
    ["RY(pi/2)", "I"],
    ["RX(pi/2)", "RY(pi/2)"],
    ["RY(pi/2)", "RX(pi/2)"],
    ["RX(pi/2)", "RY(pi)"],
    ["RY(pi/2)", "RX(pi)"],
    ["RX(pi)", "RY(pi/2)"],
    ["RY(pi)", "RX(pi/2)"],
    ["RX(pi/2)", "RX(pi)"],
    ["RX(pi)", "RX(pi/2)"],
    ["RY(pi/2)", "RY(pi)"],
    ["RY(pi)", "RY(pi/2)"],
    ["RX(pi)", "I"],
    ["RY(pi)", "I"],
    ["RX(pi/2)", "RX(pi/2)"],
    ["RY(pi/2)", "RY(pi/2)"],
]


def _acquisition(
    params: AllXYParameters,
    platform: Platform,
    qubits: Qubits,
) -> AllXYData:
    r"""
    Data acquisition for allXY experiment.
    The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the |0> state)
    is subjected to two back-to-back single-qubit gates and measured. In each round, we run 21 different gate pairs:
    ideally, the first 5 return the qubit to |0>, the next 12 drive it to superposition state, and the last 4 put the
    qubit in |1> state.
    """

    # create a Data object to store the results
    data = AllXYData()

    # sweep the parameter
    sequences = []
    ro_pulses = {}
    for gateNumber, gates in enumerate(gatelist):
        # create a sequence of pulses
        sequence = PulseSequence()
        for qubit in qubits:
            sequence, ro_pulses[qubit] = add_gate_pair_pulses_to_sequence(
                platform,
                gates,
                qubit,
                sequence,
                None,
            )

            sequences.append(sequence)

    results = platform.execute_pulse_sequences(
        sequences,
        ExecutionParameters(
            nshots=params.nshots,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )

    i = 0
    for sequence in sequences:
        for ro_pulse in sequence.ro_pulses:
            qubit = ro_pulse.qubit
            z_proj = 2 * results[ro_pulse.serial][i].probability(0) - 1
            # store the results
            data.register_qubit(qubit, z_proj, i)
        i += 1

    return data


def add_gate_pair_pulses_to_sequence(
    platform: Platform,
    gates,
    qubit,
    sequence,
    beta_param=None,
):
    pulse_duration = platform.create_RX_pulse(qubit, start=0).duration
    # All gates have equal pulse duration

    sequenceDuration = 0
    pulse_start = 0

    for gate in gates:
        if gate == "I":
            pass

        if gate == "RX(pi)":
            if beta_param == None:
                RX_pulse = platform.create_RX_pulse(
                    qubit,
                    start=pulse_start,
                )
            else:
                RX_pulse = platform.create_RX_drag_pulse(
                    qubit,
                    start=pulse_start,
                    beta=beta_param,
                )
            sequence.add(RX_pulse)

        if gate == "RX(pi/2)":
            if beta_param == None:
                RX90_pulse = platform.create_RX90_pulse(
                    qubit,
                    start=pulse_start,
                )
            else:
                RX90_pulse = platform.create_RX90_drag_pulse(
                    qubit,
                    start=pulse_start,
                    beta=beta_param,
                )
            sequence.add(RX90_pulse)

        if gate == "RY(pi)":
            if beta_param == None:
                RY_pulse = platform.create_RX_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                )
            else:
                RY_pulse = platform.create_RX_drag_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                    beta=beta_param,
                )
            sequence.add(RY_pulse)

        if gate == "RY(pi/2)":
            if beta_param == None:
                RY90_pulse = platform.create_RX90_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                )
            else:
                RY90_pulse = platform.create_RX90_drag_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                    beta=beta_param,
                )
            sequence.add(RY90_pulse)

        sequenceDuration = sequenceDuration + pulse_duration
        pulse_start = pulse_duration

    # RO pulse starting just after pair of gates
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=sequenceDuration)
    sequence.add(ro_pulse)
    return sequence, ro_pulse


allxy_unrolling = Routine(_acquisition, _fit, _plot)
"""AllXY Routine object."""
