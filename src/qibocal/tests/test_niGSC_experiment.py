import os
from shutil import rmtree

import numpy as np
import pytest
from qibo.models import Circuit

from qibocal.calibrations.niGSC.basics.circuitfactory import Qibo1qGatesFactory
from qibocal.calibrations.niGSC.basics.experiment import *


@pytest.fixture
def depths():
    return [0, 1, 5, 10, 30]


def test_Experiment_init():
    cfactory = None
    data = None
    noise_model = None
    # All None should work.
    experiment1 = Experiment(cfactory, data=data, noise_model=noise_model)
    with pytest.raises(AssertionError):
        _ = Experiment(1)
        _ = Experiment(None, 1)
        _ = Experiment(None, None, True)
        _ = Experiment(None, None, None, 1)


@pytest.mark.parametrize("nqubits", [1, 2])
@pytest.mark.parametrize("runs", [1, 3])
def test_Experiment_perform(nqubits: int, depths: list, runs: int):
    def nocircuit_dict_empty(circuit: Circuit, datadict: dict):
        datadict["nshots"] = np.random.randint(1, 10)
        return datadict

    def nocirucuit_dict_filled(circuit: Circuit, datadict: dict):
        datadict["depthstr"] = "".join((*datadict.keys(), str(datadict["depth"])))
        return datadict

    def circuit_nodata(circuit: Circuit, datadict: dict):
        datadict["depth"] = circuit.depth
        return datadict

    def circuit_data(circuit: Circuit, datadict: dict):
        nshots = datadict["nshots"]
        datadict["samples"] = circuit(nshots=nshots).samples()
        return datadict

    cfactory1 = Qibo1qGatesFactory(nqubits, depths * runs)
    checkdepths = np.array(depths * runs) + 1
    checkdepthswithstring = [f"depth{d}" for d in checkdepths]
    experiment1 = Experiment(None)
    with pytest.raises(ValueError):
        experiment1.perform(circuit_nodata)
    experiment1.circuitfactory = cfactory1
    experiment1.perform(circuit_nodata)
    assert np.allclose(experiment1.dataframe.values.flatten(), checkdepths)
    experiment1.perform(nocirucuit_dict_filled)
    assert np.array_equal(experiment1.dataframe.values[:, 0].flatten(), checkdepths)
    assert np.array_equal(
        experiment1.dataframe.values[:, 1].flatten(), checkdepthswithstring
    )
    assert experiment1.dataframe.columns[0] == "depth"
    assert experiment1.dataframe.columns[1] == "depthstr"
    cfactory2 = Qibo1qGatesFactory(nqubits, depths * runs)
    experiment2 = Experiment(cfactory2)
    experiment2.perform(nocircuit_dict_empty)
    assert experiment2.dataframe.columns[0] == "nshots"
    experiment2.perform(circuit_data)
    assert experiment2.dataframe.columns[0] == "nshots"
    assert experiment2.dataframe.columns[1] == "samples"
    for item in experiment2.data:
        assert len(item["samples"]) == item["nshots"]
        assert isinstance(item["samples"], np.ndarray)


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("qubits", [[0], [0, 1]])
def test_Experiment_save_load(nqubits: int, depths: list, runs: int, qubits: list):
    cfactory1 = Qibo1qGatesFactory(nqubits, depths * runs, qubits=qubits)
    experiment1 = Experiment(cfactory1)
    experiment1.perform(experiment1.execute)
    path1 = experiment1.save()
    experiment2 = Experiment.load(path1)
    for datarow1, datarow2 in zip(experiment1.data, experiment2.data):
        assert np.array_equal(datarow1["samples"], datarow2["samples"])
    assert experiment2.circuitfactory is None

    cfactory3 = Qibo1qGatesFactory(nqubits, depths * runs, qubits=qubits)
    experiment3 = Experiment(cfactory3)
    experiment3.prebuild()
    path3 = experiment3.save()
    experiment4 = Experiment.load(path3)
    for circuit3, circuit4 in zip(
        experiment3.circuitfactory, experiment4.circuitfactory
    ):
        assert np.array_equal(circuit3.unitary(), circuit4.unitary())

    rmtree(path1)
    rmtree(path3)
    if len(os.listdir("experiments/rb")) == 0:
        rmtree("experiments/rb")
    if len(os.listdir("experiments")) == 0:
        rmtree("experiments/")


def test_Experiment_extract():
    def populate_experiment_data(circuit: Circuit, datarow: dict) -> dict:
        datarow["rand_uniform"] = np.random.uniform(0, 1)
        datarow["rand_normal"] = np.random.uniform(0, 1)
        datarow["rand_int"] = np.random.randint(0, 2)
        return datarow

    data = [{} for _ in range(50)]
    experiment1 = Experiment(None, data=data)
    experiment1.perform(populate_experiment_data)
    randnormal_array = experiment1.extract("rand_normal")
    experiment1.extract("rand_normal", "rand_normal", "mean")
