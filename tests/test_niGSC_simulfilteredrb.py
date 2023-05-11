import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure
from qibo import gates, models

from qibocal.calibrations.niGSC import simulfilteredrb
from qibocal.calibrations.niGSC.basics import noisemodels, utils


@pytest.fixture
def depths():
    return [0, 1, 5, 10]


@pytest.fixture
def nshots():
    return 13


@pytest.mark.parametrize("nqubits", [1, 2, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("qubits", [[0], [0, 1]])
def test_experiment(nqubits: int, depths: list, runs: int, nshots: int, qubits: list):
    if max(qubits) > nqubits - 1:
        pass
    else:
        myfactory1 = simulfilteredrb.ModuleFactory(nqubits, depths * runs)
        myexperiment1 = simulfilteredrb.ModuleExperiment(myfactory1, nshots=nshots)
        myexperiment1.perform(myexperiment1.execute)
        assert isinstance(myexperiment1.data, list)
        assert isinstance(myexperiment1.data[0], dict)
        for count, datarow in enumerate(myexperiment1.data):
            assert len(datarow.keys()) == 2
            assert isinstance(datarow["samples"], np.ndarray)
            assert len(datarow["samples"]) == nshots
            assert isinstance(datarow["depth"], int)
            assert datarow["depth"] == depths[count % len(depths)]
            if not datarow["depth"]:
                assert np.array_equal(
                    datarow["samples"], np.zeros(datarow["samples"].shape)
                )
        assert isinstance(myexperiment1.dataframe, pd.DataFrame)


@pytest.mark.parametrize("nqubits", [1, 3])
@pytest.mark.parametrize("noise_params", [[0.1, 0.2, 0.1], [0.03, 0.17, 0.05]])
def test_experiment_withnoise(nqubits: int, noise_params):
    nshots = 512
    depths = [0, 5, 10]
    runs = 2
    # Build the noise model.
    noise = noisemodels.PauliErrorOnAll(*noise_params)
    # Test exectue an experiment.
    myfactory1 = simulfilteredrb.ModuleFactory(nqubits, depths * runs)
    circuit_list = list(myfactory1)
    myfaultyexperiment = simulfilteredrb.ModuleExperiment(
        circuit_list, nshots=nshots, noise_model=noise
    )
    myfaultyexperiment.perform(myfaultyexperiment.execute)
    experiment1 = simulfilteredrb.ModuleExperiment(circuit_list, nshots=nshots)
    experiment1.perform(experiment1.execute)
    experiment12 = simulfilteredrb.ModuleExperiment(circuit_list, nshots=nshots)
    experiment12.perform(experiment12.execute)
    for datarow_faulty, datarow2, datarow3 in zip(
        myfaultyexperiment.data, experiment1.data, experiment12.data
    ):
        if not datarow_faulty["depth"]:
            assert np.array_equal(datarow_faulty["samples"], datarow2["samples"])
            assert np.array_equal(datarow2["samples"], datarow3["samples"])
        else:
            probs1 = utils.probabilities(datarow_faulty["samples"])
            probs2 = utils.probabilities(datarow2["samples"])
            probs3 = utils.probabilities(datarow3["samples"])
            # Since the error channel maps the state into the maximally mixed state,
            # if the state is already maximally mixed, they will not differ.
            # So check if a probabiliy entry is zero.
            if np.any(probs2 == 0):
                assert np.linalg.norm(probs1 - probs2) > np.linalg.norm(probs2 - probs3)


def test_filterfunction():
    """Test if the filter function works, without noise."""

    nqubits = 2
    nshots = 3000
    d = 2
    # Steal the class method for calculating clifford unitaries.
    from qibo.quantum_info.random_ensembles import random_clifford

    # The first parameter is self, set it to None since it is not needed.
    g1_matrix = random_clifford(1)
    g1 = gates.Unitary(g1_matrix, 0)
    g2_matrix = random_clifford(1)
    g2 = gates.Unitary(g2_matrix, 0)
    g3_matrix = random_clifford(1)
    g3 = gates.Unitary(g3_matrix, 1)
    g4_matrix = random_clifford(1)
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
        a0 += 1 / d**2
        # lambda = (0,1)
        a1 += 1 / d * np.abs(ideal2[s[1]]) ** 2 - 1 / d**2
        # lambda = (1,0)
        a2 += 1 / d * np.abs(ideal1[s[0]]) ** 2 - 1 / d**2
        # lambda = (1,1)
        a3 += (
            np.abs(ideal1[s[0]]) ** 2 * np.abs(ideal2[s[1]]) ** 2
            - 1 / d * np.abs(ideal1[s[0]]) ** 2
            - 1 / d * np.abs(ideal2[s[1]]) ** 2
            + 1 / d**2
        )
    a0 *= 1 / (nshots)
    a1 *= (d + 1) / (nshots)
    a2 *= (d + 1) / (nshots)
    a3 *= (d + 1) ** 2 / (nshots)
    # Now do the same but with an experiment, use a list with only
    # the prebuild circuit (build it again because it was already executed).
    # No noise.
    c = models.Circuit(nqubits)
    c.add([g1, g3, g2, g4])
    c.add(gates.M(0, 1))
    experiment = simulfilteredrb.ModuleExperiment([c], nshots=nshots)
    experiment.perform(experiment.execute)
    # Compute and get the filtered signals.
    experiment.perform(simulfilteredrb.filter_function)
    # Compare the above calculated filtered signals and the signals
    # computed with the crosstalkrb method.
    assert isinstance(experiment.data, list)
    assert np.isclose(a0, experiment.data[0]["irrep0"])
    assert np.isclose(a1, experiment.data[0]["irrep1"])
    assert np.isclose(a2, experiment.data[0]["irrep2"])
    assert np.isclose(a3, experiment.data[0]["irrep3"])


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("noise_params", [[0.1, 0.1, 0.1], [0.4, 0.2, 0.01]])
def test_post_processing(
    nqubits: int, depths: list, runs: int, nshots: int, noise_params: list
):
    # Build the noise model.
    noise = noisemodels.PauliErrorOnAll(*noise_params)
    # Test exectue an experiment.
    myfactory1 = simulfilteredrb.ModuleFactory(nqubits, list(depths) * runs)
    myfaultyexperiment = simulfilteredrb.ModuleExperiment(
        myfactory1, nshots=nshots, noise_model=noise
    )
    myfaultyexperiment.perform(myfaultyexperiment.execute)
    simulfilteredrb.post_processing_sequential(myfaultyexperiment)
    aggr_df = simulfilteredrb.get_aggregational_data(myfaultyexperiment)
    assert (
        len(aggr_df) == 2**nqubits
        and aggr_df.index[0] == "irrep0"
        and aggr_df.index[1] == "irrep1"
    )
    assert "depth" in aggr_df.columns
    assert "data" in aggr_df.columns
    assert "2sigma" in aggr_df.columns
    assert "fit_func" in aggr_df.columns
    assert "popt" in aggr_df.columns
    assert "perr" in aggr_df.columns


def test_build_report():
    depths = [1, 5, 10, 15]
    nshots = 128
    runs = 10
    nqubits = 1
    noise_params = [0.01, 0.1, 0.05]
    # Build the noise model.
    noise = noisemodels.PauliErrorOnAll(*noise_params)
    # Test exectue an experiment.
    myfactory1 = simulfilteredrb.ModuleFactory(nqubits, depths * runs)
    myfaultyexperiment = simulfilteredrb.ModuleExperiment(
        myfactory1, nshots=nshots, noise_model=noise
    )
    myfaultyexperiment.perform(myfaultyexperiment.execute)
    simulfilteredrb.post_processing_sequential(myfaultyexperiment)
    aggr_df = simulfilteredrb.get_aggregational_data(myfaultyexperiment)
    report_figure, _ = simulfilteredrb.build_report(myfaultyexperiment, aggr_df)
    assert isinstance(report_figure, Figure)
