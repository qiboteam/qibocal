import numpy as np
import pytest

from qibocal.protocols.utils import guess_period


@pytest.mark.parametrize("period", [0.05, 0.08, 0.2, 0.5, 1, 1.2])
def test_guess_period(period):
    """Testing guess period function with different periods."""
    t = np.linspace(0, 1, 100, endpoint=False)
    signal = np.sin(2 * np.pi / period * t)

    extracted_period = guess_period(t, signal)
    assert (
        pytest.approx(
            extracted_period,
            rel=3e-2
            if period < 0.3
            else 20e-2
            if period < 0.8
            else 0.5
            if period < 1.5
            else None,  # raise error, window too small
        )
        == period
    )
