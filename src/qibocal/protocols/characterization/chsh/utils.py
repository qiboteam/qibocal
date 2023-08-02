"""Auxiliary functions to run CHSH protocol."""

import numpy as np

READOUT_BASIS = [("Z", "Z"), ("Z", "X"), ("X", "Z"), ("X", "X")]


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
    return chsh / nshots


def calculate_frequencies(results):
    """Calculates outcome probabilities from individual shots."""
    shots = np.stack([result.samples for result in set(results.values())]).T
    values, counts = np.unique(shots, axis=0, return_counts=True)

    return {"".join(str(i) for i in v): cnt for v, cnt in zip(values, counts)}
