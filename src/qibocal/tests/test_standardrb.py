from collections.abc import Iterable
from shutil import rmtree

import numpy as np
import pandas as pd

import pytest
from qibo import gates, models
from qibo.noise import NoiseModel, PauliError

from qibocal.calibrations.protocols import abstract, standardrb

""" FIXME the measurement branch treates measurements gates.M different
"""

@pytest.fixture
def depths():
    return [0,1,5,10,30]

@pytest.fixture
def nshots():
    return 13


@pytest.mark.parametrize("nqubits", [1, 2, 3])
@pytest.mark.parametrize("runs", [1, 3])
def test_factory(nqubits: int, depths: list, runs: int):
    """Check for how random circuits are produced and if the lengths, shape
    and randomness works.

    TODO how to check these are Cliffords?

    Args:
        qubits (list): List of qubits
        depths (list): list of depths for circuits
        runs (int): How many randomly drawn cirucit for one depth value
    """
    myfactory1 = standardrb.SingleCliffordsInvFactory(nqubits, depths, runs)
    assert isinstance(myfactory1, Iterable)
    circuits_list = list(myfactory1)
    assert len(circuits_list) == len(depths) * runs
    for count, circuit in enumerate(myfactory1):
        assert isinstance(circuit, models.Circuit)
        assert np.allclose(circuit.unitary(), np.eye(2**nqubits))
        # There will be an inverse and measurement gate, + 2.
        if circuit.ngates == 1:
            assert isinstance(circuit.queue[0], gates.measurements.M)
        else:
            assert circuit.ngates == depths[count % len(depths)] + 2
    randomnesscount = 0
    depth = depths[-1]
    circuits_list = list(standardrb.SingleCliffordsInvFactory(nqubits, [depth], runs))
    for count in range(runs - 1):
        circuit1q = circuits_list[count].queue[:depth]
        circuit2q = circuits_list[count + 1].queue[:depth]
        circuit1 = models.Circuit(nqubits)
        circuit1.add(circuit1q)
        circuit2 = models.Circuit(nqubits)
        circuit2.add(circuit2q)
        equal = np.array_equal(circuit1.unitary(), circuit2.unitary())
        randomnesscount += equal
    assert randomnesscount < runs / 2.0


@pytest.mark.parametrize("nqubits", [1, 2])
@pytest.mark.parametrize("runs", [1, 3])
def test_experiment(nqubits: int, depths: list, runs: int, nshots: int):
    """_summary_

    Args:
        qubits (list): _description_
        depths (list): _description_
        runs (int): _description_
    """
    # Test execute an experiment.
    myfactory1 = standardrb.SingleCliffordsInvFactory(nqubits, depths, runs)
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
    check_samples = np.zeros((len(depths) * runs, nshots, nqubits), dtype=int)
    assert np.array_equal(myexperiment1.samples, check_samples)
    check_probs = np.zeros((len(depths) * runs, 2**nqubits))
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

    myfactory2 = standardrb.SingleCliffordsInvFactory(nqubits, depths, runs)
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


@pytest.mark.parametrize("nqubits", [1, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("noise_params", [[0.1, 0.1, 0.1], [0.02, 0.03, 0.007]])
def test_experiment_withnoise(
    nqubits: int, depths: list, runs: int, nshots: int, noise_params: list
):
    # Build the noise model.
    paulinoise = PauliError(*noise_params)
    noise = NoiseModel()
    noise.add(paulinoise, gates.Unitary)
    # Test exectue an experiment.
    myfactory1 = standardrb.SingleCliffordsInvFactory(nqubits, depths, runs)
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
    check_samples = np.zeros((len(depths) * runs, nshots, nqubits), dtype=int)
    assert not np.array_equal(myfaultyexperiment.samples, check_samples)
    check_probs = np.zeros((len(depths) * runs, 2**nqubits))
    check_probs[:, 0] = 1.0
    assert not np.array_equal(myfaultyexperiment.probabilities, check_probs)
    assert np.array_equal(myfaultyexperiment.depths, np.tile(depths, runs))


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("qubits", [[0], [0,1]])
def test_embed_circuit(nqubits: int, depths: list, runs: int, qubits: list):
    nshots = 2
    myfactory1 = standardrb.SingleCliffordsInvFactory(
        nqubits, depths, runs, qubits=qubits
    )
    for circuit in myfactory1:
        assert circuit.nqubits == nqubits
        for gate in circuit.queue:
            # import pdb
            # pdb.set_trace()
            assert gate._target_qubits == tuple(qubits)
    myexperiment1 = standardrb.StandardRBExperiment(myfactory1, nshots)
    myexperiment1.execute()
    for samples in myexperiment1.samples:
        assert samples.shape == (nshots, len(qubits))


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
def test_analyze(nqubits: int, depths: list, runs: int, nshots: int):
    # Build the noise model.
    noise_params = [0.1, 0.1, 0.1]
    paulinoise = PauliError(*noise_params)
    noise = NoiseModel()
    noise.add(paulinoise, gates.Unitary)
    # Test exectue an experiment.
    myfactory1 = standardrb.SingleCliffordsInvFactory(nqubits, depths, runs)
    myfaultyexperiment = standardrb.StandardRBExperiment(
        myfactory1, nshots, noisemodel=noise
    )
    myfaultyexperiment.execute()

    figure = standardrb.analyze(myfaultyexperiment)
    figure.show()
