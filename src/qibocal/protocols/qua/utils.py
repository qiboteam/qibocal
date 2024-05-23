from enum import Enum
from pathlib import Path

import numpy as np

CLIFFORD = np.load(Path(__file__).parent / "qua_clifford_group.npz")


class RBType(Enum):
    STANDARD = "standard"
    FILTERED = "filtered"
    RESTLESS = "restless"

    @classmethod
    def infer(cls, apply_inverse, relaxation_time):
        if apply_inverse:
            return cls.STANDARD
        if relaxation_time < RESTLESS_RELAXATION_CUTOFF:
            assert not apply_inverse
            return cls.RESTLESS
        return cls.FILTERED


def process_data(rb_type, state, depths=None, sequences=None):
    if rb_type is RBType.STANDARD:
        return 1 - np.mean(state, axis=0), np.std(state, axis=0) / np.sqrt(
            state.shape[0]
        )

    is_restless = rb_type is RBType.RESTLESS
    term = filter_term(depths, state, sequences, is_restless=is_restless)
    ff = filter_function(term)
    return np.mean(ff, axis=1), np.std(ff, axis=1) / np.sqrt(ff.shape[1])


def power_law(power, a, b, p):
    """Function to fit to survival probability vs circuit depths."""
    return a * (p**power) + b


def clifford_mul(sequences, interleave=None):
    """Quick multiplication over single-qubit Clifford group (using integers).

    Requires the `CLIFFORD` file (loaded on top) which contains
    the group multiplication matrix and the mapping from integers to
    unitary matrices.
    The indexing is the one defined by the QUA script used for data acquisition.
    """
    group = CLIFFORD["algebra"]
    result = np.zeros_like(sequences[:, 0])
    for i in range(sequences.shape[1]):
        if interleave is not None:
            matrix = group[sequences[:, i], interleave]
        else:
            matrix = sequences[:, i]
        result = group[matrix, result]
    return result


def generate_depths(max_circuit_depth, delta_clifford):
    """Generate vector of depths compatible with the QUA acquisition script."""
    if delta_clifford == 1:
        return np.arange(1, max_circuit_depth + 0.1, delta_clifford).astype(int)
    depths = np.arange(0, max_circuit_depth + 0.1, delta_clifford).astype(int)
    depths[0] = 1
    return depths


def filter_term(depths, state, sequences, is_restless=False, interleave=None):
    """Calculate Clifford sandwich term that appears in the filter function.

    Args:
        depths (np.ndarray): Vector of depths. size: (ndepths,)
        state (np.ndarray): Matrix with acquired shots. size (num_of_sequences, ndepths)
        sequences (np.ndarray): Matrix with Clifford indices used. size (num_of_sequences, max_circuit_depth)
        interleave (int): Optional integer from 0 to 23, corresponding to the Clifford matrix to interleave.
        is_restless (bool): If `True` the restless filter function is used.
    """
    state = state.astype(int)
    seqids = np.arange(len(sequences))
    terms = []
    for i, depth in enumerate(depths):
        clifford_indices = clifford_mul(sequences[:, :depth], interleave=interleave)
        # `clifford_indices`: (num_of_sequences,)
        clifford_matrices = CLIFFORD["matrices"][clifford_indices]
        # `clifford_matrices`: (num_of_sequences, 2, 2)
        if is_restless:
            if i > 0:
                state_before = state[:, i - 1]
            else:
                state_before = np.concatenate(([0], state[:-1, i - 1]))
        else:
            state_before = 0
        terms.append(clifford_matrices[seqids, state[:, i], state_before])
    return terms


def filter_function(x):
    """Calculate filter function using Eq. (7) from notes.

    Args:
        x: Term calculated by the ``filter_term`` function above.

    Returns:
        Filter function output for each data point of shape (ndepths, num_of_sequences).
    """
    return 3 * (np.abs(x) ** 2 - 0.5)
