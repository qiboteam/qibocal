# -*- coding: utf-8 -*-
import numpy as np
from qibo import gates, get_backend


PARAMETERS = [
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


def OneQubitGate(seed=None):

    if seed is not None:
        backend = get_backend()
        backend.set_seed(seed)

    return Rn(*PARAMETERS[np.random.randint(len(PARAMETERS))])


def Rn(theta=0, nx=0, ny=0, nz=0):
    """"""
    matrix = np.array(
        [
            [
                np.cos(theta / 2) - 1.0j * nz * np.sin(theta / 2),
                -ny * np.sin(theta / 2) - 1.0j * nx * np.sin(theta / 2),
            ],
            [
                ny * np.sin(theta / 2) - 1.0j * nx * np.sin(theta / 2),
                np.cos(theta / 2) + 1.0j * nz * np.sin(theta / 2),
            ],
        ]
    )
    return gates.Unitary(matrix, 0)
