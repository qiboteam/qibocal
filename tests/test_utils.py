import numpy as np
import pytest
from pydantic import ValidationError

from qibocal.protocols.utils import cumulative, eval_magnitude, to_range


def test_cumulative():
    data = np.arange(10, dtype=float)
    assert np.array_equal(cumulative(data, data), data)


def test_eval_magnitude():
    assert 0 == eval_magnitude(0)
    assert 0 == eval_magnitude(np.inf)
    assert 3 == eval_magnitude(2000)


def test_range_conversion():
    assert to_range((0, 100, 2)) == (0, 100, 2)
    assert to_range(("linspace", 0, 100, int(1e3)))[:2] == (0, 100)
    assert pytest.approx(to_range(("window", 50, 10, 1e-1))) == (45, 55, 1e-1)
    assert pytest.approx(to_range(("linwindow", 50, 10, int(5e2)))[:2]) == (45, 55)
    assert pytest.approx(to_range(("center", 1e5, 1e-1), center=1e6)[0]) == 1e6 - 5e4
    assert pytest.approx(to_range(("lincenter", 1e5, 1234), center=1e6)[1]) == 1e6 + 5e4
    assert pytest.approx(to_range(("asym", (1e5, -2e4), 1e-1), center=1e6)[0]) == 0.9e6
    assert (
        pytest.approx(to_range(("linasym", (1e5, -2e4), 987), center=1e6)[1])
        == 1e6 - 2e4
    )

    with pytest.raises(ValueError, match="center"):
        to_range(("center", 1e6, 10))
    with pytest.raises(ValueError, match="2 steps"):
        to_range(("lincenter", 1e6, 1), center=5e9)

    with pytest.raises(ValidationError):
        to_range(("linwindow", 50, 10))
    with pytest.raises(ValidationError):
        to_range(("center", 50, 10, 20))
    with pytest.raises(ValidationError):
        to_range(("asym", -10, 10, 20))
