import numpy as np

from qibocal.protocols.characterization.utils import cumulative, eval_magnitude


def test_cumulative():
    data = np.arange(10, dtype=float)
    assert np.array_equal(cumulative(data, data), data)


def test_eval_magnitude():
    assert 0 == eval_magnitude(0)
    assert 0 == eval_magnitude(np.inf)
    assert 3 == eval_magnitude(2000)
