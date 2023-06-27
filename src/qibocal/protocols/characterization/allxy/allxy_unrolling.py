from qibolab import AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Qubits, Routine

from .allxy import (
    AllXYData,
    AllXYParameters,
    _fit,
    _plot,
    add_gate_pair_pulses_to_sequence,
    gatelist,
)


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


allxy_unrolling = Routine(_acquisition, _fit, _plot)
"""AllXY Routine object."""
