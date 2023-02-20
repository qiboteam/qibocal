from copy import deepcopy
from os import mkdir
from os.path import isdir
from typing import Union

import numpy as np
from qibo.models import Circuit

# Gates, without having to define any paramters
ONEQ_GATES = ["I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG"]


def experiment_directory(name: str):
    """Make the directory where the experiment will be stored."""
    from datetime import datetime

    overall_dir = "experiments/"
    # Check if the overall directory exists. If not create it.
    if not isdir(overall_dir):
        mkdir(overall_dir)
    # Get the current date and time.
    dt_string = datetime.now().strftime("%y%b%d_%H%M%S")
    # Every script name ``name`` gets its own directory.
    subdirectory = f"{overall_dir}{name}/"
    if not isdir(subdirectory):  # pragma: no cover
        mkdir(subdirectory)
    # Name the final directory for this experiment.
    final_directory = f"{subdirectory}experiment{dt_string}/"
    if not isdir(final_directory):  # pragma: no cover
        mkdir(final_directory)
    else:
        already_file, count = True, 1
        while already_file:
            final_directory = f"{subdirectory}experiment{dt_string}_{count}/"
            if not isdir(final_directory):
                mkdir(final_directory)
                already_file = False
            else:
                count += 1
    return final_directory


def effective_depol(error_channel, **kwargs):
    """ """
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


def copy_circuit(circuit: Circuit):
    newcircuit = Circuit(circuit.nqubits)
    for gate in circuit.queue:
        newcircuit.add(deepcopy(gate))
    return newcircuit


def gate_fidelity(eff_depol: float, primitive=False) -> float:
    """Returns the average gate fidelity given the effective depolarizing parameter for single qubits.

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


def number_to_str(number: complex):
    if np.iscomplex(number):
        the_str = "{:.2f}{:.2f}j".format(np.real(number), np.imag(number))
    else:
        the_str = "{:.3f}j".format(number)
    return the_str
