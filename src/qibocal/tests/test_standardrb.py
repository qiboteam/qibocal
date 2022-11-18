from collections.abc import Iterable
from shutil import rmtree

import numpy as np
import pandas as pd
from qibo import gates, models
from qibo.noise import NoiseModel, PauliError

from qibocal.calibrations.protocols import standardrb

""" FIXME the measurement branch treates measurements gates.M different
"""


def test_factory(qubits: list, depths: list, runs: int):
    """Check for how random circuits are produced and if the lengths, shape
    and randomness works.

    TODO how to check these are Cliffords?

    Args:
        qubits (list): List of qubits
        depths (list): list of depths for circuits
        runs (int): How many randomly drawn cirucit for one depth value
    """
    myfactory1 = standardrb.SingleCliffordsInvFactory(qubits, depths, runs)
    assert isinstance(myfactory1, Iterable)
    circuits_list = list(myfactory1)
    assert len(circuits_list) == len(depths) * runs
    for count, circuit in enumerate(myfactory1):
        assert isinstance(circuit, models.Circuit)
        assert np.allclose(circuit.unitary(), np.eye(2 ** len(qubits)))
        # There will be an inverse and measurement gate, + 2.
        assert len(circuit.queue) == depths[count % len(depths)] + 2
    randomnesscount = 0
    runs = 10
    depth = 2
    circuits_list = list(standardrb.SingleCliffordsInvFactory(qubits, [depth], runs=10))
    for count in range(runs - 1):
        circuit1q = circuits_list[count].queue[:depth]
        circuit2q = circuits_list[count + 1].queue[:depth]
        circuit1 = models.Circuit(len(qubits))
        circuit1.add(circuit1q)
        circuit2 = models.Circuit(len(qubits))
        circuit2.add(circuit2q)
        equal = np.array_equal(circuit1.unitary(), circuit2.unitary())
        randomnesscount += equal
    assert randomnesscount < runs / 2.0


def test_experiment(qubits: list, depths: list, runs: int):
    """_summary_

    Args:
        qubits (list): _description_
        depths (list): _description_
        runs (int): _description_
    """
    nshots = 10
    # Test exectue an experiment.
    myfactory1 = standardrb.SingleCliffordsInvFactory(qubits, depths, runs)
    myexperiment1 = standardrb.StandardRBExperiment(myfactory1, nshots)
    myexperiment1.execute()
    assert isinstance(myexperiment1.data, list)
    assert isinstance(myexperiment1.data[0], dict)
    for count, datarow in enumerate(myexperiment1.data):
        assert len(datarow.keys()) == 2
        assert isinstance(datarow["samples"], np.ndarray)
        assert len(datarow["samples"]) == nshots
        assert isinstance(datarow["depth"], int)
        assert datarow["depth"] == depths[count % len(depths)]
    assert isinstance(myexperiment1.dataframe, pd.DataFrame)
    check_samples = np.zeros((len(depths) * runs, nshots, len(qubits)), dtype=int)
    assert np.array_equal(myexperiment1.samples, check_samples)
    check_probs = np.zeros((len(depths) * runs, 2 ** len(qubits)))
    check_probs[:, 0] = 1.0
    assert np.array_equal(myexperiment1.probabilities, check_probs)
    assert np.array_equal(myexperiment1.depths, np.tile(depths, runs))

    myexperiment1.save()
    path1 = myexperiment1.path

    myexperiment1_loaded = standardrb.StandardRBExperiment.load(path1)
    for datarow, datarow_load in zip(myexperiment1.data, myexperiment1_loaded.data):
        assert np.array_equal(datarow["samples"], datarow_load["samples"])
        assert datarow["depth"] == datarow_load["depth"]
    assert myexperiment1_loaded.circuitfactory is None

    myfactory2 = standardrb.SingleCliffordsInvFactory(qubits, depths, runs)
    myexperiment2 = standardrb.StandardRBExperiment(myfactory2, nshots)
    assert myexperiment2.circuitfactory == myfactory2
    myexperiment2.prebuild()
    assert isinstance(myexperiment2.circuitfactory, list)
    assert len(myexperiment2.circuitfactory) == len(depths) * runs
    myexperiment2.execute()
    myexperiment2.save()
    path2 = myexperiment2.path
    myexperiment2.save()
    path3 = myexperiment2.path

    myexperiment2_loaded = standardrb.StandardRBExperiment.load(path2)
    for datarow, datarow_load in zip(myexperiment2.data, myexperiment2_loaded.data):
        assert np.array_equal(datarow["samples"], datarow_load["samples"])
        assert datarow["depth"] == datarow_load["depth"]

    for circuit, circuit_load in zip(
        myexperiment2.circuitfactory, myexperiment2_loaded.circuitfactory
    ):
        for gate, gate_load in zip(circuit.queue, circuit_load.queue):
            if not isinstance(gate, gates.M):
                m, m_load = gate.matrix, gate_load.matrix
                assert np.array_equal(m, m_load)

    rmtree(path1)
    rmtree(path2)
    rmtree(path3)


def test_experiment_withnoise(qubits: list, depths: list, runs: int):
    nshots = 100
    # Build the noise model.
    noise_params = [0.1, 0.1, 0.1]
    paulinoise = PauliError(*noise_params)
    noise = NoiseModel()
    noise.add(paulinoise, gates.Unitary)
    # Test exectue an experiment.
    myfactory1 = standardrb.SingleCliffordsInvFactory(qubits, depths, runs)
    myfaultyexperiment = standardrb.StandardRBExperiment(
        myfactory1, nshots, noisemodel=noise
    )
    myfaultyexperiment.execute()
    assert isinstance(myfaultyexperiment.data, list)
    assert isinstance(myfaultyexperiment.data[0], dict)
    for count, datarow in enumerate(myfaultyexperiment.data):
        assert len(datarow.keys()) == 2
        assert isinstance(datarow["samples"], np.ndarray)
        assert len(datarow["samples"]) == nshots
        assert isinstance(datarow["depth"], int)
        assert datarow["depth"] == depths[count % len(depths)]
    assert isinstance(myfaultyexperiment.dataframe, pd.DataFrame)
    check_samples = np.zeros((len(depths) * runs, nshots, len(qubits)), dtype=int)
    assert not np.array_equal(myfaultyexperiment.samples, check_samples)
    check_probs = np.zeros((len(depths) * runs, 2 ** len(qubits)))
    check_probs[:, 0] = 1.0
    assert not np.array_equal(myfaultyexperiment.probabilities, check_probs)
    assert np.array_equal(myfaultyexperiment.depths, np.tile(depths, runs))


def test_analyze(qubits, depths, runs):
    nshots = 100
    # Build the noise model.
    noise_params = [0.1, 0.1, 0.1]
    paulinoise = PauliError(*noise_params)
    noise = NoiseModel()
    noise.add(paulinoise, gates.Unitary)
    # Test exectue an experiment.
    myfactory1 = standardrb.SingleCliffordsInvFactory(qubits, depths, runs)
    myfaultyexperiment = standardrb.StandardRBExperiment(
        myfactory1, nshots, noisemodel=noise
    )
    myfaultyexperiment.execute()

    standardrb.analyze(myfaultyexperiment)


qubits = [0]
depths, runs = [1, 3, 4], 2

test_factory(qubits, depths, runs)
test_experiment(qubits, depths, runs)
test_experiment_withnoise(qubits, depths, runs)
test_analyze(qubits, depths, runs)
