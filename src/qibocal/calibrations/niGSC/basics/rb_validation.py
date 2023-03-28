from collections.abc import Iterable

import numpy as np
from qibo.noise import NoiseModel
from qibo.quantum_info import comp_basis_to_pauli, vectorization

from qibocal.calibrations.niGSC.basics.circuitfactory import (
    CircuitFactory,
    ZkFilteredCircuitFactory,
)
from qibocal.config import raise_error

pauli_basis_1q = comp_basis_to_pauli(1, normalize=True)


def irrep_info(circuitfactory: CircuitFactory):
    """
    Returns necessary information about irrep decomposition of the gate group.

    Args:
        circuitfactory (CircuitFactory)

    Returns:
        (basis, position, size, multiplicity) of an irrep
    """

    if circuitfactory.name == "SingleCliffords":
        return (pauli_basis_1q, 1, 3, 1)
    elif circuitfactory.name == "PauliGroup":
        return (pauli_basis_1q, 3, 1, 1)
    elif circuitfactory.name == "Id":
        return (np.eye(4), 0, 1, 4)
    elif circuitfactory.name == "XId":
        return (pauli_basis_1q, 2, 1, 2)
    elif isinstance(circuitfactory, ZkFilteredCircuitFactory):
        basis = (
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1 / np.sqrt(2), -1j / np.sqrt(2)],
                    [0, 0, 1j / np.sqrt(2), -1 / np.sqrt(2)],
                ]
            )
            @ pauli_basis_1q
        )
        index = 3
        size = 1
        multiplicity = 1
        return (basis, index, size, multiplicity)

    index = 0
    size = 4**circuitfactory.nqubits
    multiplicity = 1
    basis = np.eye(size)
    return (basis, index, size, multiplicity)


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
    basis, index, size, multiplicity = irrep_info(circuitfactory)

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
            if not ideal and noise_model is not None:
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
                    # TODO shift qubits to 0 (e.g. (1, 3) -> (0, 1))
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
                            error_channel = error.channel(0, *error.options)

                        try:
                            noise_superoperator = (
                                error_channel.to_liouville(backend=backend)
                                @ noise_superoperator
                            )
                        except:
                            print(
                                f"to_liouville() not implemented for {error_channel.name}"
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
                noise_superoperator = noisy_circuit.queue[1].to_liouville(
                    backend=backend
                )
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
    basis, index, size, multiplicity = irrep_info(circuitfactory)

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


def filtered_decay_parameters(
    circuitfactory: CircuitFactory,
    noise_model: NoiseModel,
    with_coefficients=False,
    N=None,
    backend=None,
):
    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    if not isinstance(circuitfactory, CircuitFactory):
        raise_error(
            TypeError,
            f"circuitfactory must be of type `CircuitFactory` for filtered_decay_parameters. Got {type(circuitfactory)} instead.",
        )

    # Get the ideal fourier transform onto the irrep and its spectral gap
    ideal_fourier_transform = fourier_transform(circuitfactory, None, N, True, backend)
    ideal_eigs = np.unique(np.linalg.eigvals(ideal_fourier_transform))
    ideal_eigs = np.sort(ideal_eigs)[::-1]
    spectral_gap = ideal_eigs[0] - ideal_eigs[1] if len(ideal_eigs) > 1 else 0

    # Get the noisy fourier transform onto the irrep
    noisy_fourier_transform = fourier_transform(
        circuitfactory, noise_model, N, False, backend
    )

    basis, index, size, multiplicity = irrep_info(circuitfactory)

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
