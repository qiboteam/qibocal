"""Data acquisition for randomized benchmarking."""

from collections import defaultdict
from itertools import chain

from qibolab import AveragingMode

from qibocal.auto.operation import Parameters, QubitId, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import marginalize_qubit_counts

from .circuit_generation import (
    RBGenerator,
    _execute_indexed_circuits,
    _generate_indexed_circuits,
    setup_data,
)
from .types import RB2QData, RB2QInterData, RBData, RBType


def rb_acquisition(
    params: Parameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
    inverse_layer: bool = True,
    interleave: str | None = None,
) -> RBData:
    """RB data acquisition function using explicit circuit indexing.

    Args:
        params: Experiment parameters including depths, niter, nshots, seed.
        platform: CalibrationPlatform to execute circuits on.
        targets: List of target qubit IDs.
        inverse_layer: Whether to add an inverse layer to circuits. Defaults to True.
        interleave: Interleaving pattern for circuits. Defaults to None.

    Returns:
        RBData: Validated RB data structure with results organized by (qubit, depth).
    """
    rb_gen = RBGenerator(params.seed)

    npulses_per_clifford = rb_gen.calculate_average_pulses()
    data = setup_data(
        params, npulses_per_clifford=npulses_per_clifford, single_qubit=True
    )

    indexed_circuits = _generate_indexed_circuits(
        params=params,
        rb_gen=rb_gen,
        targets=targets,
        inverse_layer=inverse_layer,
        interleave=interleave,
    )

    indexed_results = _execute_indexed_circuits(
        indexed_circuits=indexed_circuits,
        params=params,
        platform=platform,
        qubit_map=targets,
        averaging_mode=AveragingMode.CYCLIC
        if len(targets) == 1
        else AveragingMode.SINGLESHOT,
    )

    # Create a dict of the form {(qubit, depth): list[result]}.
    # This marginalises over the iterations for a given (qubit, depth)
    grouped: defaultdict = defaultdict(list)
    for indexed_result in indexed_results:
        for qubit_id, target in enumerate(targets):
            result = marginalize_qubit_counts(indexed_result.result, qubit_id)
            key = (target, indexed_result.index.depth)
            survival_counts = result["0"] if inverse_layer else result["1"]
            survival_prob = survival_counts / params.nshots
            grouped[key].append(survival_prob)

    for (qubit, depth), results in grouped.items():
        data.register_qubit(
            RBType,
            (qubit, depth),
            {"survival_probs": results},
        )

    return data


def twoq_rb_acquisition(
    params: Parameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
    inverse_layer: bool = True,
    interleave: str | None = None,
) -> RB2QData | RB2QInterData:
    """
    The data acquisition stage of two qubit Standard Randomized Benchmarking.

    Args:
        params (RB2QParameters): The parameters for the randomized benchmarking experiment.
        targets (list[QubitPairId]): The list of qubit pair IDs on which to perform the benchmarking.
        inverse_layer (bool, optional): Whether to add an inverse layer to the circuits. Defaults to True.
        interleave (str, optional): The type of interleaving to apply. Defaults to None.

    Returns:
        RB2QData: The acquired data for two qubit randomized benchmarking.
    """
    rb_gen = RBGenerator(params.seed, file=params.file)

    npulses_per_clifford = rb_gen.calculate_average_pulses()
    data = setup_data(
        params,
        npulses_per_clifford=npulses_per_clifford,
        single_qubit=False,
        interleave=interleave,
    )

    indexed_circuits = _generate_indexed_circuits(
        params=params,
        rb_gen=rb_gen,
        targets=targets,
        inverse_layer=inverse_layer,
        interleave=interleave,
    )

    indexed_results = _execute_indexed_circuits(
        indexed_circuits=indexed_circuits,
        params=params,
        platform=platform,
        qubit_map=list(chain.from_iterable(targets)),
    )

    # Create a dict of the form {(qubit[0], qubit[1], depth): list[result]}.
    # This marginalises over the iterations for a given (qubit_pair, depth)
    grouped: defaultdict = defaultdict(list)
    for indexed_result in indexed_results:
        for pair_id, qubit_pair in enumerate(targets):
            partial_result = marginalize_qubit_counts(
                indexed_result.result, [pair_id * 2, pair_id * 2 + 1]
            )
            key = (qubit_pair[0], qubit_pair[1], indexed_result.index.depth)
            survival_counts = (
                partial_result["00"] if inverse_layer else partial_result["11"]
            )
            survival_prob = survival_counts / params.nshots
            grouped[key].append(survival_prob)

    for (qubit0, qubit1, depth), results in grouped.items():
        data.register_qubit(
            dtype=RBType,
            data_keys=(qubit0, qubit1, depth),
            data_dict={"survival_probs": results},
        )

    assert isinstance(data, RB2QData | RB2QInterData)
    return data
