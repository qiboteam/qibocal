import numpy as np

from qibocal.protocols.characterization.utils import cumulative, fill_table


def test_cumulative():
    data = np.arange(10, dtype=float)
    assert np.array_equal(cumulative(data, data), data)


def test_fill_table():
    qubit = 0
    name = "foo"
    value = 10
    error = 5
    unit = "bar"
    output = f"{qubit}| {name}: (1.0 ± 0.5) 10^1 bar <br>"
    output_no_error = f"{qubit}| {name}: 1.0 * 10^1<br>"

    assert fill_table(qubit, name, value, error, unit) == output
    assert fill_table(qubit, name, value, None) == output_no_error
