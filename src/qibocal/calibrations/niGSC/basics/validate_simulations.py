import numpy as np

# TODO add fourier transform of protocol.


def inner_prod(a, b):
    # Calculate (a, b) = sum(a*ij bij)
    res = np.dot(a.conj().reshape(1, -1), b.reshape(-1, 1))
    return complex(res)


def density_to_pauli(density):
    """Returns Pauli vector of a 1-qubit density matrix"""
    s0 = np.eye(2)
    s1 = np.array([[0, 1], [1, 0]])
    s2 = np.array([[0, -1j], [1j, 0]])
    s3 = np.array([[1, 0], [0, -1]])
    paulis = np.array([s0, s1, s2, s3]) / np.sqrt(2)
    bloch_vec = []
    for i in range(0, len(paulis)):
        el = inner_prod(paulis[i], density)
        bloch_vec.append(el)
    return np.array(bloch_vec)


def partial_trace(a, n=2):
    dim = a.shape[0] // n
    res = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            for k in range(n):
                res[i][j] += a[n * i + k][n * j + k]
    return res


def get_mql(init_density=np.array([[1, 0], [0, 0]])):
    # M_sign
    m_matr = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    x_sign = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    m_sign = x_sign.T.conj() @ m_matr
    # Q_sign
    s_sign = np.array([[0, 0], [0, 1]])
    state_pauli = density_to_pauli(init_density)
    state_outer_prod = state_pauli.reshape(-1, 1) @ state_pauli.reshape(1, -1).conj()
    q_sign = state_outer_prod @ x_sign @ s_sign
    # Get tr2[|M_sign)(Q_sign|)]
    mq_sign = m_sign.reshape(-1, 1) @ q_sign.reshape(1, -1).conj()
    return partial_trace(mq_sign, 2)


def validation(fourier, init_density=np.array([[1, 0], [0, 0]])):
    # A matrix for Z-basis measurement and initial state |0><0|
    mql = get_mql(
        init_density
    )  # np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
    print(mql)
    # Get the eigenvectors and eigenvalues of the Fourier matrix s.t. r @ np.diag(eigvals) @ l == fourier
    eigvals, r = np.linalg.eig(fourier)
    l = np.linalg.inv(r)
    # Find the indices corresponing to the obtained eigenvalues in descending order
    sorted_indices = eigvals.argsort()[::-1]
    # A: Dominant part
    r1 = r[:, sorted_indices[:2]]
    l1 = l[sorted_indices[:2], :]
    a = l1 @ mql @ r1
    dom_coef = np.array([a[0][0], a[1][1]])
    dom_decays = eigvals[sorted_indices[:2]]
    # B: Subdominant part
    r2 = r[:, sorted_indices[2:]]
    l2 = l[sorted_indices[2:], :]
    b = l2 @ mql @ r2
    subdom_coef = np.array([b[0][0], b[1][1]])
    subdom_decays = eigvals[sorted_indices[2:]]
    # What should be returned?
    print(dom_coef.round(3))
    print(dom_decays.round(3))
    print(subdom_coef.round(3))
    print(subdom_decays.round(3))
    return dom_coef, dom_decays, subdom_coef, subdom_decays
