from os import mkdir
from os.path import isdir
from typing import Union

import numpy as np
from qibo import gates

# To not define the parameters for one qubit Cliffords every time a
# new qubits is drawn define the parameters as global variable.
# This are parameters for all 24 one qubit clifford gates.
ONEQUBIT_CLIFFORD_PARAMS = [
    (0, 0, 0, 0),
    (np.pi, 1, 0, 0),
    (np.pi, 0, 1, 0),
    (np.pi, 0, 0, 1),
    (np.pi / 2, 1, 0, 0),
    (-np.pi / 2, 1, 0, 0),
    (np.pi / 2, 0, 1, 0),
    (-np.pi / 2, 0, 1, 0),
    (np.pi / 2, 0, 0, 1),
    (-np.pi / 2, 0, 0, 1),
    (np.pi, 1 / np.sqrt(2), 1 / np.sqrt(2), 0),
    (np.pi, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)),
    (np.pi, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)),
    (np.pi, -1 / np.sqrt(2), 1 / np.sqrt(2), 0),
    (np.pi, 1 / np.sqrt(2), 0, -1 / np.sqrt(2)),
    (np.pi, 0, -1 / np.sqrt(2), 1 / np.sqrt(2)),
    (2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)),
]

# Gates, without having to define any paramters
ONEQ_GATES = ['I', 'X', 'Y', 'Z', 'H', 'S', 'SDG', 'T', 'TDG']

# TODO use Renatos Pauli basis.
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
pauli = [np.eye(2) / np.sqrt(2), X / np.sqrt(2), Y / np.sqrt(2), Z / np.sqrt(2)]


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
    if not isdir(subdirectory):
        mkdir(subdirectory)
    # Name the final directory for this experiment.
    final_directory = f"{subdirectory}experiment{dt_string}/"
    if not isdir(final_directory):
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


def gate_adjoint_action_to_pauli_liouville(gate: gates.gates) -> np.ndarray:
    matrix = gate.matrix
    return np.array(
        [[np.trace(p2.conj().T @ matrix @ p1 @ matrix) for p1 in pauli] for p2 in pauli]
    )

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


