from functools import reduce

import numpy as np
import pytest

from qibocal.protocols.randomized_benchmarking import fitting
from qibocal.protocols.randomized_benchmarking.dict_utils import load_inverse_cliffords
from qibocal.protocols.randomized_benchmarking.utils import (
    RB_Generator,
    generate_inv_dict_cliffords_file,
    layer_circuit,
    load_cliffords,
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


# Test utils
@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("qubits", [1, 2, [0, 1], np.array([0, 1])])
def test_random_clifford(qubits, seed):
    rb_gen = RB_Generator(seed)

    result_single = np.array([[1j, -1j], [-1j, -1j]]) / np.sqrt(2)

    result_two = np.array(
        [
            [0.0 + 0.0j, 0.5 - 0.5j, -0.0 - 0.0j, -0.5 + 0.5j],
            [0.5 + 0.5j, 0.0 + 0.0j, -0.5 - 0.5j, -0.0 - 0.0],
            [0.0 - 0.0j, -0.5 + 0.5j, 0.0 - 0.0j, -0.5 + 0.5j],
            [-0.5 - 0.5j, 0.0 - 0.0j, -0.5 - 0.5j, 0.0 - 0.0j],
        ]
    )

    result = result_single if isinstance(qubits, int) else result_two

    if isinstance(qubits, int):
        qubits = [qubits]
    gates = []
    for qubit in qubits:
        gate, index = random_clifford(rb_gen.random_index)
        gates.append(gate)

    matrix = reduce(np.kron, [gate.matrix() for gate in gates])
    assert np.allclose(matrix, result)


def test_generate_inv_dict_cliffords_file(tmp_path):
    file = "2qubitCliffs.json"
    two_qubit_cliffords = load_cliffords(file)

    tmp_path = tmp_path / "test.npz"

    clifford_inv = generate_inv_dict_cliffords_file(two_qubit_cliffords)
    np.savez(tmp_path, **clifford_inv)
    clifford_inv = np.load(tmp_path)

    file_inv = "2qubitCliffsInv.npz"
    clifford_matrices_inv = load_inverse_cliffords(file_inv)

    assert clifford_inv.files == clifford_matrices_inv.files


@pytest.mark.parametrize("depth", [1, 10, 34])
def test_layer_circuit_single_qubit(mocker, depth):
    qubit = 0
    rb_gen = RB_Generator(123)
    single_qubit_spy = mocker.spy(rb_gen, "layer_gen_single_qubit")
    two_qubit_spy = mocker.spy(rb_gen, "layer_gen_two_qubit")

    circuit, indices = layer_circuit(rb_gen, depth, qubit)

    # assert that generator was called expected number of times
    assert single_qubit_spy.call_count == depth
    assert two_qubit_spy.call_count == 0

    # assert that results from generator calls were used
    assert circuit.depth == depth
    circuit_gates = {g for m in circuit.queue.moments for g in m}
    indices = set(indices)
    for gate, i in single_qubit_spy.spy_return_list:
        assert gate in circuit_gates
        assert i in indices


@pytest.mark.parametrize("depth", [2, 24, 47])
def test_layer_circuit_two_qubit(mocker, depth):
    qubit_pair = (0, 1)
    rb_gen = RB_Generator(123, file="2qubitCliffs.json")
    single_qubit_spy = mocker.spy(rb_gen, "layer_gen_single_qubit")
    two_qubit_spy = mocker.spy(rb_gen, "layer_gen_two_qubit")

    circuit, indices = layer_circuit(rb_gen, depth, qubit_pair)

    # assert that generator was called expected number of times
    assert single_qubit_spy.call_count == 0
    assert two_qubit_spy.call_count == depth

    # assert that results from generator calls were used
    assert circuit.depth >= depth
    circuit_gates = [g for m in circuit.queue.moments for g in m if g is not None]
    indices = set(indices)
    for gates, i in two_qubit_spy.spy_return_list:
        assert all(g in circuit_gates for g in gates)
        assert i in indices
