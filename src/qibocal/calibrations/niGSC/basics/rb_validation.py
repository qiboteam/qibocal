import importlib
from collections.abc import Iterable

import numpy as np
from qibo.noise import NoiseModel
from qibo.quantum_info import comp_basis_to_pauli, vectorization

from qibocal.config import raise_error

pauli_basis_1q = comp_basis_to_pauli(1, normalize=True)


def partial_trace(a, n=1):
    if n == 1:
        return a

    dim = a.shape[0] // n
    res = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            res[i][j] = np.sum([a[n * i + k][n * j + k] for k in range(n)])
    return res


def channel_twirl(
    gates_list: list,
    nqubits: int,
    channel: np.ndarray = None,
    basis: np.ndarray = None,
    backend=None,
):
    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    basis = basis if basis is not None else np.eye(2**nqubits, dtype=complex)

    # Transform the channel to irrep's basis
    channel_transformed = basis @ channel @ basis.T.conj()

    # Compute the fourier transform onto the irrep
    twirl = np.zeros(
        (
            (4**nqubits),
            (4**nqubits),
        ),
        dtype=complex,
    )

    for gate in gates_list:
        gate_matrix = gate.asmatrix(backend)
        # Compute the superoperator of the gate
        gate_liouville = np.kron(gate_matrix, gate_matrix.conj())
        # Get the representation of the gate in the irrep's basis
        gate_representation = basis @ gate_liouville @ basis.T.conj()

        twirl += (
            gate_representation @ channel_transformed @ gate_representation.T.conj()
        )
    return twirl / len(gates_list)


def fourier_transform(
    gates_list: list,
    irrep_info: tuple,
    nqubits: int,
    noise_model: NoiseModel,
    backend=None,
):
    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    # Get the necessary information about the irreducible representation (irrep)
    basis, index, size, multiplicity = irrep_info

    basis = basis if basis is not None else np.eye(2**nqubits, dtype=complex)
    irrep_projector = basis[index : index + size, :]

    # Compute the fourier transform onto the irrep
    fourier = np.zeros(
        (
            size * (4**nqubits),
            size * (4**nqubits),
        ),
        dtype=complex,
    )

    # If N is None, compute the theoretical Fourier transform
    for gate in gates_list:
        gate_matrix = gate.asmatrix(backend)
        # Compute the superoperator of the gate
        gate_liouville = np.kron(gate_matrix, gate_matrix.conj())
        # Calculate the irrep of the gate
        gate_irrep = irrep_projector @ gate_liouville @ irrep_projector.T.conj()
        # Get the representation of the gate in the irrep's basis
        gate_representation = basis @ gate_liouville @ basis.T.conj()
        # Apply noise if the noise_model given
        if noise_model is not None:
            noise_superoperator = np.eye(4**nqubits, dtype=complex)
            # Temporary solution TODO change when channels are modified
            from qibo.noise import (
                CustomError,
                DepolarizingError,
                KrausError,
                UnitaryError,
            )

            errors_list = noise_model.errors[gate.__class__] + noise_model.errors[None]
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
    return fourier / len(gates_list)


def filtered_decay_parameters(
    module_name: str,
    nqubits: int,
    noise_model: NoiseModel,
    with_coefficients: bool = False,
    N: int = None,
    backend=None,
):
    """
    Theoretical validation of filtered RB.

    Args:
        module_name (str): name of the module in `qibocal.calibrations.niGSC` or full path to a custom module.
            Must contain ModuleFactory, irrep_info(), and gate_group() if N is None.
        nqubits (int): number of qubits.
        noise_model (:class:`qibo.noise.NoiseModel`): noise model applied to the RB circuits.
        with_coefficients (bool): returns a list of coefficients and a correponding list of exponential decays when True,
            otherwise an empty coefficients list and a list of decays. Default is False.
        N (int): number of gates sampled from a ModuleFactory when given. Default is None and the gates are taken from the module's gate_group().
        backend (:class:`qibo.backends.abstract.Backend`): Backend object to use for execution.
            If ``None`` the currently active global backend is used. Default is ``None``.
    
    Returns:
        list, list: coefficients and exponential decays
    """
    module = importlib.import_module(f"qibocal.calibrations.niGSC.{module_name}")

    # Get gate group or a list of sampled gates
    if N is None:
        gate_group = module.gate_group(nqubits)
    else:
        small_factory = module.ModuleFactory(nqubits, [1] * N)
        gate_group = [circuit.queue[0] for circuit in small_factory]

    # Get the necessary information about an irrep
    basis, index, size, multiplicity = module.irrep_info(nqubits)
    if not isinstance(basis, np.ndarray):
        raise_error(
            TypeError,
            f"basis = irrep_info[0] must be np.ndarray, got {type(basis)} instead.",
        )

    if not isinstance(index, int):
        raise_error(
            TypeError,
            f"index = irrep_info[1] must be of type int, got {type(index)} instead.",
        )

    if not isinstance(size, int):
        raise_error(
            TypeError,
            f"size = irrep_info[2] must be of type int, got {type(size)} instead.",
        )

    if not isinstance(multiplicity, int):
        raise_error(
            TypeError,
            f"multiplicity = irrep_info[3] must be of type int, got {type(multiplicity)} instead.",
        )

    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    # Get the ideal fourier transform onto the irrep and its spectral gap
    ideal_fourier_transform = fourier_transform(
        gates_list=gate_group,
        irrep_info=(basis, index, size, multiplicity),
        nqubits=nqubits,
        noise_model=None,
        backend=backend,
    )
    ideal_eigs = np.unique(np.linalg.eigvals(ideal_fourier_transform))
    ideal_eigs = np.sort(ideal_eigs)[::-1]
    spectral_gap = ideal_eigs[0] - ideal_eigs[1] if len(ideal_eigs) > 1 else 0

    # Get the noisy fourier transform onto the irrep
    noisy_fourier_transform = fourier_transform(
        gates_list=gate_group,
        irrep_info=(basis, index, size, multiplicity),
        nqubits=nqubits,
        noise_model=noise_model,
        backend=backend,
    )

    # Check if the noise is too strong and subdominant decays are not bounded
    noise_strength = np.linalg.norm(
        np.abs(noisy_fourier_transform - ideal_fourier_transform), 2
    )
    # TODO raise warning if noise is too strong and Th8 does not hold
    noise_check = noise_strength < (spectral_gap / 4)
    number_of_decays = multiplicity

    # Return the decay parameters if with_coefficients = False
    if not with_coefficients:
        decay_params = np.sort(np.linalg.eigvals(noisy_fourier_transform))[::-1]
        return [], decay_params[:number_of_decays]

    # Dephasing channel with the povm elements |i><i| where i=0...d-1
    dephasing_channel = np.zeros(
        (4**nqubits, 4**nqubits),
        dtype=complex,
    )
    for i in range(2**nqubits):
        povm_element = np.zeros((2**nqubits, 2**nqubits))
        povm_element[i][i] = 1
        dephasing_channel += (
            vectorization(povm_element).reshape(-1, 1)
            @ vectorization(povm_element).reshape(1, -1).conj()
        )

    # S = channel twirl on dephasing channel in the irrep's basis
    s_superop = channel_twirl(gate_group, nqubits, dephasing_channel, basis, backend)
    s_superop = np.linalg.pinv(s_superop)[
        index : index + size * multiplicity, index : index + size * multiplicity
    ]

    # Calculate |M_lambda)(Q_lambda|
    dephasing_channel = basis @ dephasing_channel @ basis.T.conj()
    x_lambda = np.eye(4**nqubits, dtype=complex)[
        :, index : index + size * multiplicity
    ]
    m_lambda = x_lambda.T.conj() @ dephasing_channel

    # Outer product of the initial state
    init_density = np.zeros((2**nqubits, 2**nqubits), dtype=complex)
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
