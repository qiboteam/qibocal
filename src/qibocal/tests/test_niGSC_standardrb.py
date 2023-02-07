from collections.abc import Iterable
from shutil import rmtree

import os

import numpy as np
import pandas as pd
import pytest
from qibo import gates, models
from qibo.noise import NoiseModel, PauliError
from itertools import product

from qibocal.calibrations.niGSC import standardrb
from qibocal.calibrations.niGSC.basics import utils

def theoretical_outcome(noisemodel: NoiseModel) -> float:
    """Take the used noise model acting on unitaries and calculates the
    effective depolarizing parameter.

    Args:
        experiment (Experiment): Experiment which executed the simulation.
        noisemddel (NoiseModel): Applied noise model.

    Returns:
        (float): The effective depolarizing parameter of given error.
    """

    # TODO This has to be more systematic. Delete it from the branch which will be merged.
    # Check for correctness of noise model and gate independence.
    errorkeys = noisemodel.errors.keys()
    assert len(errorkeys) == 1 and list(errorkeys)[0] == gates.Unitary
    # Extract the noise acting on unitaries and turn it into the associated
    # error channel.
    error = noisemodel.errors[gates.Unitary][0]
    errorchannel = error.channel(0, *error.options)
    # Calculate the effective depolarizing parameter.
    return utils.effective_depol(errorchannel)

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
    myfactory1 = standardrb.moduleFactory(nqubits, list(depths) * runs)
    assert isinstance(myfactory1, Iterable)
    circuits_list = list(myfactory1)
    assert len(circuits_list) == len(depths) * runs
    for count, circuit in enumerate(myfactory1):
        assert isinstance(circuit, models.Circuit)
        assert np.allclose(circuit.unitary(), np.eye(2**nqubits))
        if circuit.ngates == 1:
            assert isinstance(circuit.queue[0], gates.measurements.M)
        else:
            # There will be an inverse and measurement gate, so + 2.
            assert circuit.ngates == depths[count % len(depths)] * nqubits + 2
            assert circuit.depth == depths[count % len(depths)] + 2
    randomnesscount = 0
    depth = depths[-1]
    circuits_list = list(standardrb.moduleFactory(nqubits, [depth] * runs))
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
    myfactory1 = standardrb.moduleFactory(nqubits, list(depths) * runs)
    myexperiment1 = standardrb.moduleExperiment(myfactory1, nshots)
    myexperiment1.perform(myexperiment1.execute)
    assert isinstance(myexperiment1.data, list)
    assert isinstance(myexperiment1.data[0], dict)
    for count, datarow in enumerate(myexperiment1.data):
        assert len(datarow.keys()) == 2
        assert isinstance(datarow["samples"], np.ndarray)
        assert len(datarow["samples"]) == nshots
        assert isinstance(datarow["depth"], int)
        assert datarow["depth"] == depths[count % len(depths)]
    assert isinstance(myexperiment1.dataframe, pd.DataFrame)


    myexperiment1.save()
    path1 = myexperiment1.path

    myexperiment1_loaded = standardrb.moduleExperiment.load(path1)
    for datarow, datarow_load in zip(myexperiment1.data, myexperiment1_loaded.data):
        assert np.array_equal(datarow["samples"], datarow_load["samples"])
        assert datarow["depth"] == datarow_load["depth"]
    assert myexperiment1_loaded.circuitfactory is None

    myfactory2 = standardrb.moduleFactory(nqubits, depths, runs)
    myexperiment2 = standardrb.moduleExperiment(myfactory2, nshots)
    assert myexperiment2.circuitfactory == myfactory2
    myexperiment2.prebuild()
    assert isinstance(myexperiment2.circuitfactory, list)
    assert len(myexperiment2.circuitfactory) == len(depths) * runs
    myexperiment2.perform(myexperiment2.execute)
    # TODO this only works when the .copy() method works.
    # myexperiment2.save()
    # path2 = myexperiment2.path
    # myexperiment2.save()
    # path3 = myexperiment2.path

    # myexperiment2_loaded = standardrb.moduleExperiment.load(path2)
    # for datarow, datarow_load in zip(myexperiment2.data, myexperiment2_loaded.data):
    #     assert np.array_equal(datarow["samples"], datarow_load["samples"])
    #     assert datarow["depth"] == datarow_load["depth"]

    # for circuit, circuit_load in zip(
    #     myexperiment2.circuitfactory, myexperiment2_loaded.circuitfactory
    # ):
    #     for gate, gate_load in zip(circuit.queue, circuit_load.queue):
    #         if not isinstance(gate, gates.M):
    #             m, m_load = gate.matrix, gate_load.matrix
    #             assert np.array_equal(m, m_load)

    rmtree(path1)
    if len(os.listdir('experiments/rb')) == 0:
        rmtree('experiments/rb')
    if len(os.listdir('experiments')) == 0:
        rmtree('experiments/')
        
    # rmtree(path2)
    # rmtree(path3)


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
    myfactory1 = standardrb.moduleFactory(nqubits, list(depths) * runs)
    myfaultyexperiment = standardrb.moduleExperiment(
        myfactory1, nshots, noisemodel=noise
    )
    myfaultyexperiment.perform(myfaultyexperiment.execute)
    assert isinstance(myfaultyexperiment.data, list)
    assert isinstance(myfaultyexperiment.data[0], dict)
    for count, datarow in enumerate(myfaultyexperiment.data):
        assert len(datarow.keys()) == 2
        assert isinstance(datarow["samples"], np.ndarray)
        assert len(datarow["samples"]) == nshots
        assert isinstance(datarow["depth"], int)
        assert datarow["depth"] == depths[count % len(depths)]
    assert isinstance(myfaultyexperiment.dataframe, pd.DataFrame)


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("qubits", [[0], [0, 1]])
def test_embed_circuit(nqubits: int, depths: list, runs: int, qubits: list):
    nshots = 2
    myfactory1 = standardrb.moduleFactory(
        nqubits, list(depths) * runs, qubits=qubits
    )
    test_list = list(product(qubits))
    test_list.append(tuple(qubits))
    for circuit in myfactory1:
        assert circuit.nqubits == nqubits
        for gate in circuit.queue:
            # import pdb
            # pdb.set_trace()
            assert gate._target_qubits in test_list
    myexperiment1 = standardrb.moduleExperiment(myfactory1, nshots)
    myexperiment1.perform(myexperiment1.execute)


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
def test_utils_probs(nqubits: int, depths: list, runs: int, nshots: int):

    # Build the noise model.
    noise_params = [0.0001, 0.001, 0.0005]
    paulinoise = PauliError(*noise_params)
    noise = NoiseModel()
    noise.add(paulinoise, gates.Unitary)
    # Test exectue an experiment.
    myfactory1 = standardrb.moduleFactory(nqubits, list(depths) * runs)
    myfaultyexperiment = standardrb.moduleExperiment(
        myfactory1, nshots, noisemodel=noise
    )
    myfaultyexperiment.perform(myfaultyexperiment.execute)
    myfaultyexperiment.perform(standardrb.groundstate_probabilities)
    probs = utils.probabilities(myfaultyexperiment.extract('samples'))
    assert probs.shape == (runs * len(depths), 2 ** nqubits)
    assert np.allclose(np.sum(probs, axis=1), 1)
    for probsarray in probs:
        if probsarray[0] < 1.:
            assert np.all(np.greater_equal(probsarray[0] * np.ones(len(probsarray)), probsarray))
        else:
            assert probsarray[0] == 1.


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
def test_post_processing(nqubits: int, depths: list, runs: int, nshots: int):

    # Build the noise model.
    noise_params = [0.01, 0.3, 0.14]
    paulinoise = PauliError(*noise_params)
    noise = NoiseModel()
    noise.add(paulinoise, gates.Unitary)
    # Test exectue an experiment.
    myfactory1 = standardrb.moduleFactory(nqubits, list(depths) * runs)
    myfaultyexperiment = standardrb.moduleExperiment(
        myfactory1, nshots, noisemodel=noise
    )
    myfaultyexperiment.perform(myfaultyexperiment.execute)
    standardrb.post_processing_sequential(myfaultyexperiment)
    probs = utils.probabilities(myfaultyexperiment.extract('samples'))
    ground_probs = probs[:, 0]
    test_ground_probs = myfaultyexperiment.extract('groundstate probability')
    assert np.allclose(ground_probs, test_ground_probs)
    aggr_df = standardrb.get_aggregational_data(myfaultyexperiment)
    assert len(aggr_df) == 1 and aggr_df.index[0] == 'groundstate probability'
    assert 'depth' in aggr_df.columns
    assert 'data' in aggr_df.columns
    assert '2sigma' in aggr_df.columns
    assert 'fit_func' in aggr_df.columns
    assert 'popt' in aggr_df.columns
    assert 'perr' in aggr_df.columns


def test_build_report():
    depths = [1, 5, 10, 15, 20, 25]
    nshots = 128
    runs = 10
    nqubits = 1
    # Build the noise model.
    noise_params = [0.01, 0.1, 0.05]
    paulinoise = PauliError(*noise_params)
    noise = NoiseModel()
    noise.add(paulinoise, gates.Unitary)
    # Test exectue an experiment.
    myfactory1 = standardrb.moduleFactory(nqubits, depths * runs)
    myfaultyexperiment = standardrb.moduleExperiment(
        myfactory1, nshots, noisemodel=noise
    )
    myfaultyexperiment.perform(myfaultyexperiment.execute)
    standardrb.post_processing_sequential(myfaultyexperiment)
    aggr_df = standardrb.get_aggregational_data(myfaultyexperiment)
    assert theoretical_outcome(noise) - aggr_df.popt[0]['p'] < 2 * aggr_df.perr[0]['p_err']
    figure = standardrb.build_report(myfaultyexperiment, aggr_df)


