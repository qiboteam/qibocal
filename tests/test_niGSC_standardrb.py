from itertools import product

import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure
from qibo import gates
from qibo.noise import NoiseModel

from qibocal.calibrations.niGSC import standardrb
from qibocal.calibrations.niGSC.basics import noisemodels, utils


def theoretical_outcome(noise_model: NoiseModel) -> float:
    """Take the used noise model acting on unitaries and calculates the
    effective depolarizing parameter.

    Args:
        experiment (Experiment): Experiment which executed the simulation.
        noisemodel (NoiseModel): Applied noise model.

    Returns:
        (float): The effective depolarizing parameter of given error.
    """

    # TODO This has to be more systematic. Delete it from the branch which will be merged.
    # Check for correctness of noise model and gate independence.
    assert None in noise_model.errors
    # Extract the noise acting on unitaries and turn it into the associated
    # error channel.
    error = noise_model.errors[None][0][1]
    errorchannel = error.channel(0, *error.options)
    # Calculate the effective depolarizing parameter.
    return utils.effective_depol(errorchannel)


@pytest.fixture
def depths():
    return [0, 1, 5, 10]


@pytest.fixture
def nshots():
    return 127


@pytest.mark.parametrize("nqubits", [1, 2])
@pytest.mark.parametrize("runs", [1, 3])
def test_experiment(nqubits: int, depths: list, runs: int, nshots: int):
    # Test execute an experiment.
    myfactory1 = standardrb.ModuleFactory(nqubits, list(depths) * runs)
    myexperiment1 = standardrb.ModuleExperiment(myfactory1, nshots=nshots)
    assert myexperiment1.name == "StandardRB"
    myexperiment1.perform(myexperiment1.execute)
    assert isinstance(myexperiment1.data, list)
    assert isinstance(myexperiment1.data[0], dict)
    for count, datarow in enumerate(myexperiment1.data):
        assert len(datarow.keys()) == 2
        assert isinstance(datarow["samples"], np.ndarray)
        assert len(datarow["samples"]) == nshots
        assert isinstance(datarow["depth"], int)
        assert datarow["depth"] == depths[count % len(depths)]
        assert np.array_equal(datarow["samples"], np.zeros(datarow["samples"].shape))
    assert isinstance(myexperiment1.dataframe, pd.DataFrame)


@pytest.mark.parametrize("nqubits", [1, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("noise_params", [[0.1, 0.1, 0.1], [0.02, 0.3, 0.07]])
def test_experiment_withnoise(
    nqubits: int, depths: list, runs: int, nshots: int, noise_params: list
):
    # Build the noise model.
    noise = noisemodels.PauliErrorOnAll(*noise_params)
    # Test exectue an experiment.
    myfactory1 = standardrb.ModuleFactory(nqubits, list(depths) * runs)
    myfaultyexperiment = standardrb.ModuleExperiment(
        myfactory1, nshots=nshots, noise_model=noise
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
        # If there are no executed gates (other than the measurement) or only one
        # the probability that no error occured is too high.
        if datarow["depth"] > 2:
            assert not np.array_equal(
                datarow["samples"], np.zeros(datarow["samples"].shape)
            )
    assert isinstance(myfaultyexperiment.dataframe, pd.DataFrame)


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("qubits", [[0], [0, 1]])
def test_embed_circuit(nqubits: int, depths: list, runs: int, qubits: list):
    nshots = 2
    myfactory1 = standardrb.ModuleFactory(nqubits, list(depths) * runs, qubits=qubits)
    test_list = list(product(qubits))
    test_list.append(tuple(qubits))
    for circuit in myfactory1:
        assert circuit.nqubits == nqubits
        for gate in circuit.queue:
            assert gate._target_qubits in test_list
    myexperiment1 = standardrb.ModuleExperiment(myfactory1, nshots=nshots)
    myexperiment1.perform(myexperiment1.execute)


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
def test_utils_probs_and_noisy_execution(
    nqubits: int, depths: list, runs: int, nshots: int
):
    noise_params = [0.0001, 0.001, 0.0005]
    # Build the noise model.
    noise = noisemodels.PauliErrorOnAll(*noise_params)
    # Test exectue an experiment.
    myfactory1 = standardrb.ModuleFactory(nqubits, list(depths) * runs)
    myfaultyexperiment = standardrb.ModuleExperiment(
        myfactory1, nshots=nshots, noise_model=noise
    )
    myfaultyexperiment.perform(myfaultyexperiment.execute)
    myfaultyexperiment.perform(standardrb.groundstate_probabilities)
    probs = utils.probabilities(myfaultyexperiment.extract("samples"))
    assert probs.shape == (runs * len(depths), 2**nqubits)
    assert np.allclose(np.sum(probs, axis=1), 1)
    for probsarray in probs:
        if probsarray[0] < 1.0:
            assert np.all(
                np.greater_equal(probsarray[0] * np.ones(len(probsarray)), probsarray)
            )
        else:
            assert probsarray[0] == 1.0


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
def test_post_processing(nqubits: int, depths: list, runs: int, nshots: int):
    noise_params = [0.01, 0.3, 0.14]
    # Build the noise model.
    noise = noisemodels.PauliErrorOnAll(*noise_params)
    # Test exectue an experiment.
    myfactory1 = standardrb.ModuleFactory(nqubits, list(depths) * runs)
    myfaultyexperiment = standardrb.ModuleExperiment(
        myfactory1, nshots=nshots, noise_model=noise
    )
    myfaultyexperiment.perform(myfaultyexperiment.execute)
    standardrb.post_processing_sequential(myfaultyexperiment)
    probs = utils.probabilities(myfaultyexperiment.extract("samples"))
    ground_probs = probs[:, 0]
    test_ground_probs = myfaultyexperiment.extract("groundstate probability")
    assert np.allclose(ground_probs, test_ground_probs)
    aggr_df = standardrb.get_aggregational_data(myfaultyexperiment)
    assert len(aggr_df) == 1 and aggr_df.index[0] == "groundstate probability"
    assert "depth" in aggr_df.columns
    assert "data" in aggr_df.columns
    assert "2sigma" in aggr_df.columns
    assert "fit_func" in aggr_df.columns
    assert "popt" in aggr_df.columns
    assert "perr" in aggr_df.columns


def test_build_report():
    depths = [1, 5, 10]
    nshots = 1024
    runs = 5
    nqubits = 1
    noise_params = [0.01, 0.1, 0.05]
    # Build the noise model.
    noise = noisemodels.PauliErrorOnAll(*noise_params)
    # Test exectue an experiment.
    myfactory1 = standardrb.ModuleFactory(nqubits, depths * runs)
    myfaultyexperiment = standardrb.ModuleExperiment(
        myfactory1, nshots=nshots, noise_model=noise
    )
    myfaultyexperiment.perform(myfaultyexperiment.execute)
    standardrb.post_processing_sequential(myfaultyexperiment)
    aggr_df = standardrb.get_aggregational_data(myfaultyexperiment)
    assert (
        theoretical_outcome(noise) - aggr_df.popt[0]["p"]
        < 2 * aggr_df.perr[0]["p_err"] + theoretical_outcome(noise) * 0.05
    )
    report_figure, _ = standardrb.build_report(myfaultyexperiment, aggr_df)
    assert isinstance(report_figure, Figure)
