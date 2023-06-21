from functools import reduce

import numpy as np
import pytest
import qibo

from qibocal.protocols.characterization.randomized_benchmarking import (
    fitting,
    noisemodels,
)
from qibocal.protocols.characterization.randomized_benchmarking.utils import (
    extract_from_data,
    number_to_str,
    random_clifford,
)


# Test fitting
def test_1expfitting():
    successes = 0
    number_runs = 50
    for _ in range(number_runs):
        x = np.sort(np.random.choice(np.linspace(0, 30, 50), size=20, replace=False))
        A, B = np.random.uniform(0.1, 0.99, size=2)
        f = np.random.uniform(0.4, 0.99)
        y = A * f**x + B
        assert np.allclose(fitting.exp1B_func(x, A, f, B), y)
        # Distort ``y`` a bit.
        y_dist = y + np.random.randn(len(y)) * 0.005
        popt, perr = fitting.fit_exp1B_func(x, y_dist)
        successes += np.all(
            np.abs(np.array(popt) - [A, f, B])
            < 2 * np.array(perr) + 0.05 * np.array([A, f, B]),
        )
    assert successes >= number_runs * 0.8

    successes = 0
    number_runs = 50
    for _ in range(number_runs):
        x = np.sort(np.random.choice(np.linspace(0, 30, 50), size=20, replace=False))
        A = np.random.uniform(0.1, 0.99)
        f = np.random.uniform(0.4, 0.99)
        y = A * f**x
        # Distort ``y`` a bit.
        y_dist = y + np.random.randn(len(y)) * 0.005
        popt, perr = fitting.fit_exp1_func(x, y_dist)
        successes += np.all(
            np.abs(np.array(popt) - [A, f])
            < 2 * np.array(perr) + 0.05 * np.array([A, f])
        )
    assert successes >= number_runs * 0.8

    x = np.sort(np.random.choice(np.linspace(-5, 5, 50), size=20, replace=False))
    y = np.zeros(len(x)) + 0.75
    assert np.array_equal(
        np.array(fitting.fit_exp1B_func(x, y)), np.array(((0.75, 1.0, 0), (0, 0, 0)))
    )
    assert np.array_equal(
        np.array(fitting.fit_exp1_func(x, y)), np.array(((0.75, 1.0), (0, 0)))
    )
    # Catch exceptions
    x = np.linspace(-np.pi / 2, np.pi / 2, 100)
    y_dist = np.tan(x)
    popt, perr = fitting.fit_exp1_func(x, y_dist, maxfev=1, p0=[1])
    assert not (np.all(np.array([*popt, *perr]), 0))
    popt, perr = fitting.fit_exp1B_func(x, y_dist, maxfev=1)
    assert not (np.all(np.array([*popt, *perr]), 0))


def test_exp2_fitting():
    successes = 0
    number_runs = 50
    for count in range(number_runs):
        x = np.arange(0, 50)
        A1, A2 = np.random.uniform(0.1, 0.99, size=2)
        if not count % 3:
            f1, f2 = np.random.uniform(0.1, 0.5, size=2) * 1j + np.random.uniform(
                0.1, 0.99, size=2
            )
        else:
            f1, f2 = np.random.uniform(0.1, 0.99, size=2)
        y = A1 * f1**x + A2 * f2**x
        assert np.allclose(fitting.exp2_func(x, A1, A2, f1, f2), y)
        # Distort ``y`` a bit.
        y_dist = y + np.random.uniform(-1, 1, size=len(y)) * 0.001
        popt, perr = fitting.fit_exp2_func(x, y_dist)
        worked = np.all(
            np.logical_or(
                np.allclose(np.array(popt), [A2, A1, f2, f1], atol=0.05, rtol=0.1),
                np.allclose(np.array(popt), [A1, A2, f1, f2], atol=0.05, rtol=0.1),
            )
        )
        if not worked:
            np.allclose(
                popt[0] * popt[2] ** x + popt[1] * popt[3] ** x,
                y_dist,
                atol=0.01,
                rtol=0.01,
            )
            worked = True
        successes += worked
    # This is a pretty bad rate. The ESPRIT algorithm has to be optimized.
    assert successes >= number_runs * 0.4

    with pytest.raises(ValueError):
        x = np.array([1, 2, 3, 5])
        A1, A2, f1, f2 = np.random.uniform(0.1, 0.99, size=4)
        y = A1 * f1**x + A2 * f2**x
        # Distort ``y`` a bit.
        y_dist = y + np.random.uniform(-1, 1, size=len(y)) * 0.001
        popt, perr = fitting.fit_exp2_func(x, y_dist)


# Test noisemodels
def test_PauliErrors():
    def test_model(noise_model, num_keys=1):
        assert isinstance(noise_model, qibo.noise.NoiseModel)
        errorkeys = noise_model.errors.keys()
        assert len(errorkeys) == num_keys
        error = list(noise_model.errors.values())[0][0][1]
        assert isinstance(error, qibo.noise.PauliError)
        assert len(error.options) == 3 and np.sum(pair[1] for pair in error.options) < 1

    noise_model1 = noisemodels.PauliErrorOnAll()
    test_model(noise_model1)
    noise_model2 = noisemodels.PauliErrorOnAll([0.1, 0.1, 0.1])
    test_model(noise_model2)
    noise_model3 = noisemodels.PauliErrorOnAll(None)
    test_model(noise_model3)
    with pytest.raises(ValueError):
        noise_model4 = noisemodels.PauliErrorOnAll([0.1, 0.2])

    noise_model1 = noisemodels.PauliErrorOnX()
    test_model(noise_model1)
    noise_model2 = noisemodels.PauliErrorOnX([0.1, 0.1, 0.1])
    test_model(noise_model2)
    noise_model3 = noisemodels.PauliErrorOnX(None)
    test_model(noise_model3)
    with pytest.raises(ValueError):
        noise_model4 = noisemodels.PauliErrorOnX([0.1, 0.2])


# Test utils
@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("qubits", [1, 2, [0, 1], np.array([0, 1])])
def test_random_clifford(qubits, seed):
    with pytest.raises(TypeError):
        q = "1"
        random_clifford(q)
    with pytest.raises(ValueError):
        q = -1
        random_clifford(q)
    with pytest.raises(ValueError):
        q = [0, 1, -3]
        random_clifford(q)

    result_single = np.array([[1j, -1j], [-1j, -1j]]) / np.sqrt(2)

    result_two = np.array(
        [
            [0.0 + 0.0j, 0.5 - 0.5j, -0.0 - 0.0j, -0.5 + 0.5j],
            [0.5 + 0.5j, 0.0 + 0.0j, -0.5 - 0.5j, -0.0 - 0.0],
            [0.0 - 0.0j, -0.5 + 0.5j, 0.0 - 0.0j, -0.5 + 0.5j],
            [-0.5 - 0.5j, 0.0 - 0.0j, -0.5 - 0.5j, 0.0 - 0.0j],
        ]
    )

    result = result_single if (isinstance(qubits, int) and qubits == 1) else result_two

    gates = random_clifford(qubits, seed=seed)
    matrix = reduce(np.kron, [gate.matrix for gate in gates])
    assert np.allclose(matrix, result)


@pytest.mark.parametrize("prec", [2, 3])
def test_number_to_str(prec):
    # Real values
    value = np.random.uniform(0, 1)
    assert number_to_str(value, precision=prec) == f"{value:.{prec}f}"
    assert number_to_str(value, [None, None], prec) == f"{value:.{prec}f}"
    assert number_to_str(value, 0.0123, prec) == f"{value:.3f} \u00B1 0.012"
    assert number_to_str(value, [0.0123, 0.012], prec) == f"{value:.3f} \u00B1 0.012"
    assert (
        number_to_str(value, [0.2, 1e-5], prec) == f"{value:.{prec}f} +1.0e-05 / -0.20"
    )
    assert number_to_str(value, [np.inf] * 2, prec) == f"{value:.{prec}f} \u00B1 inf"
    # Complex values
    value += np.random.uniform(0, 1) * 1j
    assert number_to_str(value, precision=prec) == f"{value:.{prec}f}"
    assert number_to_str(value, [None, None], prec) == f"{value:.{prec}f}"
    assert (
        number_to_str(value, 0.0123, prec)
        == f"({np.real(value):.3f}+{np.imag(value):.{prec}f}j) \u00B1 (0.012+0j)"
    )
    assert (
        number_to_str(value, [0.2, 1e-5], prec)
        == f"({value:.{prec}f}) +(1.0e-05+0j) / -(0.20+0j)"
    )
    assert number_to_str(value, [np.inf] * 2, prec) == f"{value:.{prec}f} \u00B1 inf"


def test_extract_from_data():
    data = [
        {"group": 1, "output": 3},
        {"group": 1, "output": 4},
        {"group": 2, "output": 5},
    ]
    assert np.allclose(extract_from_data(data, "output"), [3, 4, 5])
    assert extract_from_data(data, "output", agg_type="count") == 3
    assert np.allclose(
        extract_from_data(data, "output", "group"), ([1, 1, 2], [3, 4, 5])
    )
    assert np.allclose(
        extract_from_data(data, "output", "group", "count"), ([1, 2], [2, 1])
    )
