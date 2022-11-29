from collections.abc import Iterable
from shutil import rmtree

import numpy as np
import pandas as pd
import pytest
from qibo import gates, models
from qibo.noise import NoiseModel, PauliError

from qibocal.calibrations.protocols import crosstalkrb


@pytest.fixture
def depths():
    return [0, 1, 5, 10, 30]


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
    myfactory1 = crosstalkrb.SingleCliffordsFactory(nqubits, depths, runs)
    assert isinstance(myfactory1, Iterable)
    circuits_list = list(myfactory1)
    assert len(circuits_list) == len(depths) * runs
    for count, circuit in enumerate(myfactory1):
        assert isinstance(circuit, models.Circuit)
        assert np.allclose(circuit.unitary(), np.eye(2**nqubits))
        # There will be a measurement gate, + 1.
        if circuit.ngates == 1:
            assert isinstance(circuit.queue[0], gates.measurements.M)
        else:
            assert circuit.ngates == depths[count % len(depths)] * nqubits + 1
    randomnesscount = 0
    depth = depths[-1]
    circuits_list = list(crosstalkrb.SingleCliffordsFactory(nqubits, [depth], runs))
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
    """Test the experiment class from crosstalkrb module.

    Args:
        qubits (list): _description_
        depths (list): _description_
        runs (int): _description_
    """
    # Test execute an experiment.
    myfactory1 = crosstalkrb.SingleCliffordsFactory(nqubits, depths, runs)
    myexperiment1 = crosstalkrb.CrosstalkRBExperiment(myfactory1, nshots)
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

    myexperiment1_loaded = crosstalkrb.CrosstalkRBExperiment.load(path1)
    for datarow, datarow_load in zip(myexperiment1.data, myexperiment1_loaded.data):
        assert np.array_equal(datarow["samples"], datarow_load["samples"])
        assert datarow["depth"] == datarow_load["depth"]
    assert myexperiment1_loaded.circuitfactory is None

    myfactory2 = crosstalkrb.SingleCliffordsFactory(nqubits, depths, runs)
    myexperiment2 = crosstalkrb.CrosstalkRBExperiment(myfactory2, nshots)
    assert myexperiment2.circuitfactory == myfactory2
    myexperiment2.prebuild()
    assert isinstance(myexperiment2.circuitfactory, list)
    assert len(myexperiment2.circuitfactory) == len(depths) * runs
    myexperiment2.execute()
    myexperiment2.save()
    path2 = myexperiment2.path
    myexperiment2.save()
    path3 = myexperiment2.path

    myexperiment2_loaded = crosstalkrb.CrosstalkRBExperiment.load(path2)
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
    myfactory1 = crosstalkrb.SingleCliffordsFactory(nqubits, depths, runs)
    myfaultyexperiment = crosstalkrb.CrosstalkRBExperiment(
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


def test_filterfunction():
    """Test if the filter function works, without noise."""
    from qibocal.calibrations.protocols.utils import ONEQUBIT_CLIFFORD_PARAMS

    nqubits = 2
    nshots = 3000
    d = 2
    # Steal the class method for calculating clifford unitaries.
    clifford_unitary = crosstalkrb.SingleCliffordsFactory.clifford_unitary
    # The first parameter is self, set it to None since it is not needed.
    g1_matrix = clifford_unitary(None, *ONEQUBIT_CLIFFORD_PARAMS[8])
    g1 = gates.Unitary(g1_matrix, 0)
    g2_matrix = clifford_unitary(None, *ONEQUBIT_CLIFFORD_PARAMS[6])
    g2 = gates.Unitary(g2_matrix, 0)
    g3_matrix = clifford_unitary(None, *ONEQUBIT_CLIFFORD_PARAMS[2])
    g3 = gates.Unitary(g3_matrix, 1)
    g4_matrix = clifford_unitary(None, *ONEQUBIT_CLIFFORD_PARAMS[23])
    g4 = gates.Unitary(g4_matrix, 1)
    # Calculate the ideal unitary and the ideal outcomes.
    g21 = g2_matrix @ g1_matrix
    g43 = g4_matrix @ g3_matrix
    ideal1 = g21 @ np.array([[1], [0]])
    ideal2 = g43 @ np.array([[1], [0]])
    # Build the circuit with the ideal unitaries.
    c = models.Circuit(nqubits)
    c.add([g1, g3, g2, g4])
    c.add(gates.M(0, 1))
    # Execute the circuit and get the samples.
    samples = c(nshots=nshots).samples()
    # Initiate the variables to store the four irrep signals.
    a0, a1, a2, a3 = 0, 0, 0, 0
    for s in samples:
        # lambda = (0,0)
        a0 += 1
        # lambda = (1,0)
        a1 += d * np.abs(ideal1[s[0]]) - 1
        # lambda = (0,1)
        a2 += d * np.abs(ideal2[s[1]]) - 1
        # lambda = (1,1)
        a3 += (
            d**2 * np.abs(ideal1[s[0]]) * np.abs(ideal2[s[1]])
            - d * np.abs(ideal1[s[0]])
            - d * np.abs(ideal2[s[1]])
            + 1
        )
    a0 *= (d + 1) / (d**2 * nshots)
    a1 *= (d + 1) / (d**2 * nshots)
    a2 *= (d + 1) / (d**2 * nshots)
    a3 *= (d + 1) / (d**2 * nshots)
    # Now do the same but with an experiment, use a list with only
    # the prebuild circuit (build it again because it was already executed).
    # No noise.
    c = models.Circuit(nqubits)
    c.add([g1, g3, g2, g4])
    c.add(gates.M(0, 1))
    experiment = crosstalkrb.CrosstalkRBExperiment([c], nshots)
    experiment.execute()
    # Compute and get the filtered signals.
    experiment.apply_task(crosstalkrb.filter_function)
    list_crosstalk = experiment.data[0]["crosstalk"]
    # Compare the above calculated filtered signals and the signals
    # computed with the crosstalkrb method.
    assert np.isclose(a0, list_crosstalk[0])
    assert np.isclose(a1, list_crosstalk[1])
    assert np.isclose(a2, list_crosstalk[2])
    assert np.isclose(a3, list_crosstalk[3])


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("noise_params", [[0.1, 0.1, 0.1], [0.4, 0.2, 0.01]])
def test_analyze(
    nqubits: int, depths: list, runs: int, nshots: int, noise_params: list
):
    # Build the noise model.
    paulinoise = PauliError(*noise_params)
    noise = NoiseModel()
    noise.add(paulinoise, gates.Unitary)
    # Test exectue an experiment.
    myfactory1 = crosstalkrb.SingleCliffordsFactory(nqubits, depths, runs)
    myfaultyexperiment = crosstalkrb.CrosstalkRBExperiment(
        myfactory1, nshots, noisemodel=noise
    )
    myfaultyexperiment.execute()
    crosstalkrb.analyze(myfaultyexperiment).show()
