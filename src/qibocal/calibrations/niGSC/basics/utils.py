from typing import Union

import numpy as np

# Gates, without having to define any paramters
ONEQ_GATES = ["I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG"]


def effective_depol(error_channel, **kwargs):
    """Computes the effective depolarizing error of a channel.

    Args:
        error_channel (qibo.gates.Channel): Noise channel with `to_pauli_liouville` representation.

    Returns:
        float: Effective depolarizing error of a given error_channel.
    """
    liouvillerep = error_channel.to_pauli_liouville(normalize=True)
    d = int(np.sqrt(len(liouvillerep)))
    depolp = (np.trace(liouvillerep) - 1) / (d**2 - 1)
    return depolp


def probabilities(allsamples: Union[list, np.ndarray]) -> np.ndarray:
    """Takes the given list/array (3-dimensional) of samples and returns probabilities
    for each possible state to occure.

    The states for 4 qubits are order as follows:
    [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0),
    (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1),
    (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]

    Args:
        allsamples (Union[list, np.ndarray]): The single shot samples, 3-dimensional.

    Returns:
        np.ndarray: Probability array of 2 dimension.
    """

    from itertools import product

    # Make it an array to use the shape property.
    allsamples = np.array(allsamples)
    # The array has to have three dimension.
    if len(allsamples.shape) == 2:
        allsamples = allsamples[None, ...]
    nqubits, nshots = len(allsamples[0][0]), len(allsamples[0])
    # Create all possible state vectors.
    allstates = list(product([0, 1], repeat=nqubits))
    # Iterate over all the samples and count the different states.
    probs = [
        [np.sum(np.product(samples == state, axis=1)) for state in allstates]
        for samples in allsamples
    ]
    probs = np.array(probs) / (nshots)
    return probs


def gate_fidelity(eff_depol: float, primitive=False) -> float:
    """Returns the average gate fidelity given the effective
    depolarizing parameter for single qubits.

    If primitive is True, divide by additional 1.875 as convetion in RB reporting.
    (The original reasoning was that Clifford gates are typically
    compiled with an average number of 1.875 Pi half pulses.)

    Args:
        eff_depol (float): The effective depolarizing parameter.
        primitive (bool, optional): If True, additionally divide by 1.875.

    Returns:
        float: Average gate fidelity
    """
    infidelity = (1 - eff_depol) / 2
    if primitive:
        infidelity /= 1.875
    return 1 - infidelity


def number_to_str(number: Union[int, float, complex]) -> str:
    """Converts a number into a string.
    Returns only the real number if imaginary part is < 1e-3.

    Args:
        number (int | float | complex)

    Returns:
        str: The number expressed as a string, with three floating points.
    """
    return (
        f"{np.real(number):.3f}" if np.abs(np.imag(number)) < 1e-3 else f"{number:.3f}"
    )
