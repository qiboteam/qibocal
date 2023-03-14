from collections.abc import Iterable
from copy import deepcopy
from os import mkdir
from os.path import isdir
from typing import Union

import numpy as np
from qibo import matrices
from qibo.models import Circuit
from qibo.noise import NoiseModel
from qibo.quantum_info import vectorization

# Gates, without having to define any paramters
ONEQ_GATES = ["I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG"]
ONEQ_GATES_MATRICES = {
    "I": matrices.I,
    "X": matrices.X,
    "Y": matrices.Y,
    "Z": matrices.Z,
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]]),
    "SDG": np.array([[1, 0], [0, -1j]]),
    "T": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]]),
    "TDG": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
}


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


def copy_circuit(circuit: Circuit) -> Circuit:
    """Truly deepcopies a given circuit by copying the gates.


    Right now, qibos copy function changes properties of the copied circuit.

    Args:
        circuit (Circuit): The circuit which is copied.

    Returns:
        (Circuit): The copied circuit
    """
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


def number_to_str(number: Union[int, float, complex]) -> str:
    """Converts a number into a string.

    Necessary when storing a complex number in JASON format.

    Args:
        number (int | float | complex)

    Returns:
        str: The number expressed as a string, with two floating points when
        complex or three when real.
    """
    if np.abs(np.imag(number)) > 1e-4:
        the_str = "{:.2f}{}{:.2f}j".format(
            np.real(number),
            "+" if np.imag(number) >= 0 else "-",
            np.abs(np.imag(number)),
        )
    else:
        the_str = (
            "{:.3f}".format(np.real(number)) if np.abs(np.real(number)) > 1e-4 else "0"
        )
    return the_str


def fourier_transform(
    circuitfactory: Iterable,
    noise_model: NoiseModel,
    N: int = None,
    ideal=False,
    backend=None,
):
    if not isinstance(circuitfactory, Iterable):
        raise_error(
            TypeError,
            f"given circuit factory has wrong type {type(circuitfactory)}, must be CircuitFactory.",
        )

    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    # Get the necessary information about the irreducible representation (irrep)
    basis, index, size, multiplicity = circuitfactory.irrep_info()

    basis = (
        basis
        if basis is not None
        else np.eye(2**circuitfactory.nqubits, dtype=complex)
    )
    size = size if size is not None else basis.shape[0]
    index = index if index is not None else 0
    irrep_projector = basis[index : index + size, :]

    # Compute the fourier transform onto the irrep
    fourier = np.zeros(
        (
            size * (4**circuitfactory.nqubits),
            size * (4**circuitfactory.nqubits),
        ),
        dtype=complex,
    )

    # If N is None, compute the theoretical Fourier transform
    if N is None:
        gate_group = circuitfactory.gate_group()
        for gate in gate_group:
            gate_matrix = gate.asmatrix(backend)
            # Compute the superoperator of the gate
            gate_liouville = np.kron(gate_matrix, gate_matrix.conj())
            # Calculate the irrep of the gate
            gate_irrep = irrep_projector @ gate_liouville @ irrep_projector.T.conj()
            # Get the representation of the gate in the irrep's basis
            gate_representation = basis @ gate_liouville @ basis.T.conj()
            # Apply noise
            if (
                not ideal
                and noise_model is not None
                and gate.__class__ in noise_model.errors
            ):
                noise_superoperator = np.eye(4**circuitfactory.nqubits, dtype=complex)
                # Temporary solution TODO change when channels are modified
                from qibo.noise import (
                    CustomError,
                    DepolarizingError,
                    KrausError,
                    UnitaryError,
                )

                errors_list = (
                    noise_model.errors[gate.__class__] + noise_model.errors[None]
                )
                for condition, error, qubits in errors_list:
                    qubits = (
                        gate.qubits
                        if qubits is None
                        else tuple(set(gate.qubits) & set(qubits))
                    )
                    if condition is None or condition(gate):
                        if isinstance(error, CustomError) and qubits:
                            error_channel = error.channel
                        elif isinstance(error, DepolarizingError) and qubits:
                            error_channel = error.channel(qubits, *error.options)
                        elif isinstance(error, UnitaryError) or isinstance(
                            error, KrausError
                        ):
                            if error.rank == 2:
                                for q in qubits:
                                    error_channel = error.channel([q])
                            elif error.rank == 2 ** len(qubits):
                                error_channel = error.channel(qubits)
                        else:
                            error_channel = error.channel(qubits[0], *error.options)

                        try:
                            noise_superoperator = (
                                error_channel.to_superop(backend=backend)
                                @ noise_superoperator
                            )
                        except:
                            print(
                                f"to_superop() not implemented for {error_channel.name}"
                            )

                # Transform obtained superoperator to irrep's basis
                noise_superoperator = basis @ noise_superoperator @ basis.T.conj()
                gate_representation = noise_superoperator @ gate_representation
            fourier += np.kron(np.conj(gate_irrep), gate_representation)
        return fourier / len(gate_group)

    small_factory = circuitfactory.__class__(circuitfactory.nqubits, [1] * N)

    for circuit in small_factory:
        # Get the gate from the ideal circuit
        gate = circuit.unitary()
        # Compute the superoperator of the gate
        gate_liouville = np.kron(gate, gate.conj())
        # Get the representation of the gate in the irrep's basis
        gate_representation = basis @ gate_liouville @ basis.T.conj()
        # Calculate the irrep of the gate
        gate_irrep = irrep_projector @ gate_liouville @ irrep_projector.T.conj()
        # Apply noise
        if not ideal and noise_model is not None:
            noisy_circuit = noise_model.apply(circuit)
            if noisy_circuit.depth > circuit.depth:
                # noise_superoperator = noisy_circuit.queue[1].to_superop(backend)
                noise_superoperator = noisy_circuit.queue[1].to_superop(backend=backend)
                noise_superoperator = basis @ noise_superoperator @ basis.T.conj()
                gate_representation = noise_superoperator @ gate_representation
        fourier += np.kron(np.conj(gate_irrep), gate_representation)

    return fourier / N


def channel_twirl(
    circuitfactory: Iterable,
    channel: np.ndarray = None,
    N: int = None,
    backend=None,
):
    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    # Get the necessary information about the irreducible representation (irrep)
    basis, index, size, multiplicity = circuitfactory.irrep_info()

    basis = (
        basis
        if basis is not None
        else np.eye(2**circuitfactory.nqubits, dtype=complex)
    )

    # Transform the channel to irrep's basis
    channel_transformed = basis @ channel @ basis.T.conj()

    # Compute the fourier transform onto the irrep
    twirl = np.zeros(
        (
            (4**circuitfactory.nqubits),
            (4**circuitfactory.nqubits),
        ),
        dtype=complex,
    )

    # If N is None, compute theoretically with Haar measure
    if N is None:
        gate_group = circuitfactory.gate_group()
        for gate in gate_group:
            gate_matrix = gate.asmatrix(backend)
            # Compute the superoperator of the gate
            gate_liouville = np.kron(gate_matrix, gate_matrix.conj())
            # Get the representation of the gate in the irrep's basis
            gate_representation = basis @ gate_liouville @ basis.T.conj()

            twirl += (
                gate_representation @ channel_transformed @ gate_representation.T.conj()
            )
        return twirl / len(gate_group)

    # If N is not None, sample gates from the circuitfactory
    small_factory = circuitfactory.__class__(circuitfactory.nqubits, [1] * N)

    for circuit in small_factory:
        # Get the gate from the ideal circuit
        gate = circuit.unitary()
        # Compute the superoperator of the gate
        gate_liouville = np.kron(gate, gate.conj())
        # Get the representation of the gate in the irrep's basis
        gate_representation = basis @ gate_liouville @ basis.T.conj()
        #
        twirl += (
            gate_representation @ channel_transformed @ gate_representation.T.conj()
        )

    return twirl / N


def partial_trace(a, n=1):
    if n == 1:
        return a

    dim = a.shape[0] // n
    res = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            res[i][j] = np.sum([a[n * i + k][n * j + k] for k in range(n)])
    return res


def filtered_rb_validation(
    circuitfactory: Iterable,
    noise_model: NoiseModel,
    with_coefficients=False,
    N=None,
    backend=None,
):
    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    # Get the ideal fourier transform onto the irrep and its spectral gap
    ideal_fourier_transform = fourier_transform(circuitfactory, None, N, True, backend)
    ideal_eigs = np.unique(np.linalg.eigvals(ideal_fourier_transform))
    ideal_eigs = np.abs(np.sort(ideal_eigs)[::-1])
    spectral_gap = ideal_eigs[0] - ideal_eigs[1] if len(ideal_eigs) > 1 else 0

    # Get the noisy fourier transform onto the irrep
    noisy_fourier_transform = fourier_transform(
        circuitfactory, noise_model, N, False, backend
    )

    basis, index, size, multiplicity = circuitfactory.irrep_info()

    # Check if the noise is too strong and subdominant decays are not bounded
    noise_strength = np.linalg.norm(
        np.abs(noisy_fourier_transform - ideal_fourier_transform), 2
    )
    noise_check = noise_strength < (spectral_gap / 4)
    number_of_decays = multiplicity if noise_check else noisy_fourier_transform.shape[0]

    # Return the decay parameters if with_coefficients = False
    if not with_coefficients:
        decay_params = np.sort(np.linalg.eigvals(noisy_fourier_transform))[::-1]
        return decay_params[:number_of_decays]

    # Dephasing channel with the povm elements |i><i| where i=0...d-1
    dephasing_channel = np.zeros(
        (4**circuitfactory.nqubits, 4**circuitfactory.nqubits),
        dtype=complex,
    )
    for i in range(2**circuitfactory.nqubits):
        povm_element = np.zeros(
            (2**circuitfactory.nqubits, 2**circuitfactory.nqubits)
        )
        povm_element[i][i] = 1
        dephasing_channel += (
            vectorization(povm_element).reshape(-1, 1)
            @ vectorization(povm_element).reshape(1, -1).conj()
        )

    # S = channel twirl on dephasing channel in the irrep's basis
    s_superop = channel_twirl(circuitfactory, dephasing_channel, N)
    s_superop = np.linalg.pinv(s_superop)[
        index : index + size * multiplicity, index : index + size * multiplicity
    ]

    # Calculate |M_lambda)(Q_lambda|
    dephasing_channel = basis @ dephasing_channel @ basis.T.conj()
    x_lambda = np.eye(4**circuitfactory.nqubits, dtype=complex)[
        :, index : index + size * multiplicity
    ]
    m_lambda = x_lambda.T.conj() @ dephasing_channel

    # Outer product of the initial state
    init_density = np.zeros(
        (2**circuitfactory.nqubits, 2**circuitfactory.nqubits), dtype=complex
    )
    init_density[0][0] = 1
    state_transformed = basis @ vectorization(init_density).reshape(-1, 1)
    state_outer_prod = (
        state_transformed.reshape(-1, 1) @ state_transformed.reshape(1, -1).conj()
    )
    q_lambda = (state_outer_prod @ x_lambda @ s_superop).T.conj()

    mq_lambda = m_lambda.reshape(-1, 1) @ q_lambda.reshape(1, -1).conj()
    mq_lambda = partial_trace(mq_lambda, multiplicity)

    # Eigenvectors and eigenvalues of the Fourier matrix
    eigvals, r = np.linalg.eig(noisy_fourier_transform)
    l = np.linalg.inv(r)
    # Indices corresponing to the obtained eigenvalues in descending order
    sorted_indices = np.abs(eigvals).argsort()[::-1]
    sorted_indices = sorted_indices[:number_of_decays]
    # Coefficients corresponding to the dominant eigenvalues
    r1 = r[:, sorted_indices]
    l1 = l[sorted_indices, :]
    coeffs = np.diag(l1 @ mq_lambda @ r1)
    decays = eigvals[sorted_indices]
    # Sort the coefficients
    sorted_indices = np.abs(coeffs).argsort()[::-1]
    coeffs = coeffs[sorted_indices]
    decays = decays[sorted_indices]

    return coeffs, decays
