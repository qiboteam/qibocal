import numpy as np

from qibocal.protocols.utils import eval_magnitude


def test_eval_magnitude():
    assert 0 == eval_magnitude(0)
    assert 0 == eval_magnitude(np.inf)
    assert 3 == eval_magnitude(2000)
