from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y


def compute_mermin(frequencies, mermin_coefficients):
    """Computes the chsh inequality out of the frequencies of the 4 circuits executed."""
    assert len(frequencies) == len(mermin_coefficients)
    m = 0
    for j, freq in enumerate(frequencies):
        for key in freq:
            m += (
                mermin_coefficients[j]
                * freq[key]
                * (-1) ** (sum([int(key[k]) for k in range(len(key))]))
            )
    nshots = sum(freq[x] for x in freq)
    if nshots != 0:
        return float(m / nshots)

    return 0


def get_mermin_polynomial(n):
    assert n > 1
    m0 = X(0)
    m0p = Y(0)
    for i in range(1, n):
        mn = m0 * (X(i) + Y(i)) + m0p * (X(i) - Y(i))
        mnp = m0 * (Y(i) - X(i)) + m0p * (X(i) + Y(i))
        m0 = mn.expand()
        m0p = mnp.expand()
    m = m0 / 2 ** ((n - 1) // 2)
    return SymbolicHamiltonian(m.expand())


def get_readout_basis(mermin_polynomial):
    return [
        "".join([factor.name[0] for factor in term.factors])
        for term in mermin_polynomial.terms
    ]


def get_mermin_coefficients(mermin_polynomial):
    return [term.coefficient.real for term in mermin_polynomial.terms]
