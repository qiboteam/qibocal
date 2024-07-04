import numpy as np
import pytest

from qibocal.protocols.utils import guess_period


@pytest.mark.parametrize("period", [0.2, 0.5, 1])
def test_guess_period(period):
    """Testing guess period function with different periods."""
    t = np.linspace(0, 1, 100, endpoint=False)
    signal = np.sin(2 * np.pi / period * t)

    extracted_period = guess_period(t, signal)
    np.testing.assert_allclose(period, extracted_period)
