from os import mkdir
from os.path import isdir

import numpy as np
from qibo.models import Circuit
from qibo.noise import PauliError

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


def liouville_representation_errorchannel(error_channel, **kwargs):
    """For single qubit error channels only."""
    # For single qubit the dimension is two.
    if isinstance(error_channel, PauliError):
        flipprobs = error_channel.options

        def acts(gmatrix):
            return (
                (1 - flipprobs[0] - flipprobs[1] - flipprobs[2]) * gmatrix
                + flipprobs[0] * X @ gmatrix @ X
                + flipprobs[1] * Y @ gmatrix @ Y
                + flipprobs[2] * Z @ gmatrix @ Z
            )

    return np.array(
        [[np.trace(p2.conj().T @ acts(p1)) for p1 in pauli] for p2 in pauli]
    )


def effective_depol(error_channel, **kwargs):
    """ """
    liouvillerep = liouville_representation_errorchannel(error_channel)
    d = int(np.sqrt(len(liouvillerep)))
    depolp = ((np.trace(liouvillerep) + d) / (d + 1) - 1) / (d - 1)
    return depolp


def embed_unitary_circuit(circuit: Circuit, nqubits: int, support: list) -> Circuit:
    """Takes a circuit and redistributes the gates to the support of
    a new circuit with ``nqubits`` qubits.

    Args:
        circuit (Circuit): The circuit with len(``support``) many qubits.
        nqubits (int): Qubits of new circuit.
        support (list): The qubits were the gates should be places.

    Returns:
        Circuit: Circuit with redistributed gates.
    """

    idxmap = np.vectorize(lambda idx: support[idx])
    newcircuit = Circuit(nqubits)
    for gate in circuit.queue:
        if not isinstance(gate, gates.measurements.M):
            newcircuit.add(
                gate.__class__(gate.init_args[0], *idxmap(np.array(gate.init_args[1:])))
            )
        else:
            newcircuit.add(gates.M(*idxmap(np.array(gate.init_args[0:]))))
    return newcircuit
