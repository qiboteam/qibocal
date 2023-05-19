import numpy as np


def calculate_frequencies(result1, result2):
    """Calculates two-qubit outcome probabilities from individual shots."""
    shots = np.stack([result1.shots, result2.shots]).T
    values, counts = np.unique(shots, axis=0, return_counts=True)
    nshots = np.sum(counts)
    return {f"{int(v1)}{int(v2)}": cnt for (v1, v2), cnt in zip(values, counts)}


def calculate_probabilities(result1, result2):
    """Calculates two-qubit outcome probabilities from individual shots."""
    shots = np.stack([result1.shots, result2.shots]).T
    values, counts = np.unique(shots, axis=0, return_counts=True)
    nshots = np.sum(counts)
    return {
        f"{int(v1)}{int(v2)}": cnt / nshots for (v1, v2), cnt in zip(values, counts)
    }