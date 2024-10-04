import numpy as np
from qibolab import create_platform

from qibocal.protocols.flux_dependence.utils import (
    frequency_to_bias,
    transmon_frequency,
)
from qibocal.protocols.utils import HZ_TO_GHZ

PLATFORM = create_platform("dummy")
QUBITS = [0, 1]


def test_frequency_to_bias():
    """Test frequency to bias conversion."""

    # populate crosstalk matrix in dummy
    for i in QUBITS:
        for j in QUBITS:
            PLATFORM.qubits[i].crosstalk_matrix[j] = (
                np.random.rand() if i == j else np.random.rand() * 1e-3
            )

    # define target frequencies
    target_freqs = {
        0: PLATFORM.qubits[0].drive_frequency * HZ_TO_GHZ - 0.2,
        1: PLATFORM.qubits[0].drive_frequency * HZ_TO_GHZ + 0.2,
    }

    # expected biases
    biases = frequency_to_bias(target_freqs, PLATFORM)

    freq_q1 = transmon_frequency(
        xi=biases[0],
        xj=0,
        d=0,
        w_max=PLATFORM.qubits[0].drive_frequency * HZ_TO_GHZ,
        normalization=PLATFORM.qubits[0].crosstalk_matrix[
            0
        ],  # because M_1 - m12 m21 / M2
        offset=-PLATFORM.qubits[0].sweetspot,
        crosstalk_element=PLATFORM.qubits[0].crosstalk_matrix[1],
        charging_energy=-PLATFORM.qubits[0].anharmonicity * HZ_TO_GHZ,
    )

    freq_q2 = transmon_frequency(
        xi=biases[1],
        xj=0,
        d=0,
        w_max=PLATFORM.qubits[1].drive_frequency * HZ_TO_GHZ,
        normalization=PLATFORM.qubits[1].crosstalk_matrix[1],
        offset=-PLATFORM.qubits[1].sweetspot,
        crosstalk_element=PLATFORM.qubits[1].crosstalk_matrix[0],
        charging_energy=-PLATFORM.qubits[1].anharmonicity * HZ_TO_GHZ,
    )
    np.testing.assert_allclose(freq_q1, target_freqs[0], rtol=1e-3)
    np.testing.assert_allclose(freq_q2, target_freqs[1], rtol=1e-3)
