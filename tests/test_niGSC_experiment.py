import os
from shutil import rmtree

import numpy as np
import pytest
from qibo.models import Circuit

from qibocal.calibrations.niGSC.basics.circuitfactory import Qibo1qGatesFactory
from qibocal.calibrations.niGSC.basics.experiment import *
from qibocal.calibrations.niGSC.basics.noisemodels import PauliErrorOnX


@pytest.fixture
def depths():
    return [0, 1, 5, 10, 30]


def test_Experiment_init():
    cfactory = None
    data = None
    noise_model = None
    # All None should work.
    experiment = Experiment(cfactory, data=data, noise_model=noise_model)
    assert experiment.name == "Abstract"
    with pytest.raises(TypeError):
        Experiment(1)
        Experiment(None, 1)
        Experiment(None, None, True)
        Experiment(None, None, None, 1)


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("qubits", [[0], [0, 1]])
@pytest.mark.parametrize("nshots", [12, 23])
def test_Experiment_execute(
    nqubits: int, depths: list, runs: int, qubits: list, nshots: int
):
    cfactory1 = Qibo1qGatesFactory(nqubits, depths * runs, qubits=qubits)
    experiment1 = Experiment(cfactory1, nshots=nshots)
    experiment1.perform(experiment1.execute)
    assert experiment1.extract("samples").shape == (
        len(depths * runs),
        nshots,
        len(qubits),
    )
    noise_model = PauliErrorOnX()
    cfactory2 = Qibo1qGatesFactory(nqubits, depths * runs, qubits=qubits)
    experiment2 = Experiment(cfactory2, nshots=nshots, noise_model=noise_model)
    experiment2.perform(experiment2.execute)
    assert experiment2.extract("samples").shape == (
        len(depths * runs),
        nshots,
        len(qubits),
    )


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
    path = "_test_rb"
    if not os.path.exists(path):
        os.makedirs(path)
    path1 = experiment1.save(path)
    experiment2 = Experiment.load(path1)
    for datarow1, datarow2 in zip(experiment1.data, experiment2.data):
        assert np.array_equal(datarow1["samples"], datarow2["samples"])
    assert experiment2.circuitfactory is None
    rmtree(path)

    cfactory3 = Qibo1qGatesFactory(nqubits, depths * runs, qubits=qubits)
    experiment3 = Experiment(cfactory3)
    experiment3.prebuild()
    path = "_test_rb_1"
    if not os.path.exists(path):
        os.makedirs(path)
    path3 = experiment3.save_circuits(path)
    path3 = experiment3.save(path)
    experiment4 = Experiment.load(path3)
    for circuit3, circuit4 in zip(
        experiment3.circuitfactory, experiment4.circuitfactory
    ):
        assert np.array_equal(circuit3.unitary(), circuit4.unitary())
    rmtree(path)

    cfactory5 = Qibo1qGatesFactory(nqubits, depths * runs, qubits=qubits)
    experiment5 = Experiment(cfactory5)
    experiment5.prebuild()
    path = "_test_rb_2"
    if not os.path.exists(path):
        os.makedirs(path)
    path5 = experiment5.save_circuits(path)
    path5 = experiment5.save(path)

    experiment6 = Experiment.load(path5)
    assert experiment6.data is None
    for circuit5, circuit6 in zip(
        experiment5.circuitfactory, experiment6.circuitfactory
    ):
        assert np.array_equal(circuit5.unitary(), circuit6.unitary())
    rmtree(path)


@pytest.mark.parametrize("amount_data", [50, 71])
def test_Experiment_extract(amount_data):
    def populate_experiment_data(circuit: Circuit, datarow: dict) -> dict:
        datarow["rand_uniform"] = np.random.uniform(0, 1)
        datarow["rand_normal"] = np.random.uniform(0, 1)
        datarow["rand_int"] = np.random.randint(0, 2) + 10
        return datarow

    data = [{} for _ in range(amount_data)]
    experiment1 = Experiment(None, data=data)
    experiment1.perform(populate_experiment_data)
    # No group key and not agg_type.
    randnormal_array = experiment1.extract("rand_normal")
    # No groupkey but agg_type.
    assert np.mean(randnormal_array) == experiment1.extract("rand_normal", "", "mean")
    # No agg_type but group key.
    rand_ints, rand_uniforms = experiment1.extract(
        "rand_uniform", groupby_key="rand_int"
    )
    indx_randint = np.argsort(experiment1.dataframe["rand_int"].to_numpy())
    randints_tocheck = experiment1.dataframe["rand_int"].to_numpy()[indx_randint]
    assert np.array_equal(randints_tocheck, rand_ints)
    assert len(np.unique(rand_ints)) == 2
    until = np.count_nonzero(randints_tocheck == rand_ints[0])
    a_bit_sorted_uniforms = experiment1.dataframe["rand_uniform"].to_numpy()[
        indx_randint
    ]
    each_randint_uniforms = [rand_uniforms[:until], rand_uniforms[until:]]
    each_randint_uniforms_check = [
        a_bit_sorted_uniforms[:until],
        a_bit_sorted_uniforms[until:],
    ]
    # Since they does not neccessarily have to be ordered the same way:
    for array1, array2 in zip(each_randint_uniforms, each_randint_uniforms_check):
        for element in array1:
            assert element in array2
    # Group key and agg_type.
    rand_ints, mean_uniforms = experiment1.extract("rand_uniform", "rand_int", "mean")
    assert np.isclose(mean_uniforms[0], np.mean(each_randint_uniforms_check[0]))
    assert np.isclose(mean_uniforms[1], np.mean(each_randint_uniforms_check[1]))
