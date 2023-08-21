import numpy as np

from qibocal.protocols.characterization.utils import cumulative


def test_cumulative():
    data = np.arange(10, dtype=float)
    assert np.array_equal(cumulative(data, data), data)
