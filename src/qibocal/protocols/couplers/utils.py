import numpy as np


def chevron_function(t, delta, g):
    return (
        1
        - 4
        * g**2
        / (4 * g**2 + delta**2)
        * np.sin(np.sqrt(delta**2 + 4 * g**2) * t / 2) ** 2
    )
