import pytest

from qibocal.protocols.characterization.randomized_benchmarking.utils import (
    number_to_str,
)


@pytest.mark.parametrize("value", [0.555555, 2, -0.1 + 0.1j])
def test_number_to_str(value):
    assert number_to_str(value) == f"{value:.3f}"
    assert number_to_str(value, [None, None]) == f"{value:.3f}"
    assert number_to_str(value, 0.01) == f"{value:.2f} \u00B1 0.01"
    assert number_to_str(value, [0.01, 0.01]) == f"{value:.2f} \u00B1 0.01"
    assert number_to_str(value, [0.2, 0.01]) == f"{value:.2f} +0.01 / -0.20"
    assert (
        number_to_str(value, [float("inf"), float("inf")]) == f"{value:.3f} \u00B1 inf"
    )
    with pytest.raises(TypeError):
        test = number_to_str(value, precision="1")
    with pytest.raises(ValueError):
        test = number_to_str(value, precision=-1)
    with pytest.raises(TypeError):
        test = number_to_str(value, uncertainty="0.1")
    with pytest.raises(ValueError):
        test = number_to_str(value, uncertainty=[0.1, 0.1, 0.1])
