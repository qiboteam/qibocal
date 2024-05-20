"""Auxiliary functions to run CHSH protocol."""

from qibo.config import log

READOUT_BASIS = ["ZZ", "ZX", "XZ", "XX"]


def compute_chsh(frequencies, basis, i):
    """Computes the chsh inequality out of the frequencies of the 4 circuits executed."""
    chsh = 0
    aux = 0
    for freq in frequencies:
        for outcome in freq:
            if aux == 1 + 2 * (
                basis % 2
            ):  # This value sets where the minus sign is in the CHSH inequality
                chsh -= (-1) ** (int(outcome[0]) + int(outcome[1])) * freq[outcome][i]
            else:
                chsh += (-1) ** (int(outcome[0]) + int(outcome[1])) * freq[outcome][i]
        aux += 1
    nshots = sum(freq[x][i] for x in freq)
    try:
        return chsh / nshots
    except ZeroDivisionError:
        log.warning("Zero number of shots, returning zero.")
        return 0
