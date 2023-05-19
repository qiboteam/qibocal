import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure

from qibocal.calibrations.niGSC import XIdrb
from qibocal.calibrations.niGSC.basics import noisemodels


@pytest.fixture
def depths():
    return [0, 1, 2, 3, 4, 5]


@pytest.fixture
def nshots():
    return 27


@pytest.mark.parametrize("nqubits", [1, 2])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("qubits", [[0], [1]])
def test_experiment(nqubits: int, depths: list, runs: int, nshots: int, qubits: list):
    if max(qubits) > nqubits - 1:
        qubits = [0, 1]
        with pytest.raises(ValueError):
            myfactory1 = XIdrb.ModuleFactory(2, list(depths) * runs, qubits)
    else:
        myfactory1 = XIdrb.ModuleFactory(nqubits, list(depths) * runs, qubits)
        myexperiment1 = XIdrb.ModuleExperiment(myfactory1, nshots=nshots)
        assert myexperiment1.name == "XIdRB"
        myexperiment1.perform(myexperiment1.execute)
        assert isinstance(myexperiment1.data, list)
        assert isinstance(myexperiment1.data[0], dict)
        for count, datarow in enumerate(myexperiment1.data):
            assert len(datarow.keys()) == 3
            assert isinstance(datarow["samples"], np.ndarray)
            assert len(datarow["samples"]) == nshots
            assert isinstance(datarow["depth"], int)
            assert datarow["depth"] == depths[count % len(depths)]
            assert np.array_equal(
                datarow["samples"],
                np.zeros(datarow["samples"].shape) + datarow["countX"] % 2,
            )
        assert isinstance(myexperiment1.dataframe, pd.DataFrame)


@pytest.mark.parametrize("nqubits", [1, 3])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("noise_params", [[0.1, 0.1, 0.1], [0.02, 0.3, 0.07]])
@pytest.mark.parametrize("qubits", [[0], [1]])
def test_experiment_withnoise(
    nqubits: int, depths: list, runs: int, qubits: list, noise_params: list
):
    nshots = 100
    if max(qubits) > nqubits - 1:
        pass
    else:
        # Build the noise model.
        noise = noisemodels.PauliErrorOnX(*noise_params)
        # Test exectue an experiment.
        myfactory1 = XIdrb.ModuleFactory(nqubits, list(depths) * runs, qubits)
        myfaultyexperiment = XIdrb.ModuleExperiment(
            myfactory1, nshots=nshots, noise_model=noise
        )
        myfaultyexperiment.perform(myfaultyexperiment.execute)
        assert isinstance(myfaultyexperiment.data, list)
        assert isinstance(myfaultyexperiment.data[0], dict)
        for count, datarow in enumerate(myfaultyexperiment.data):
            assert len(datarow.keys()) == 3
            assert isinstance(datarow["samples"], np.ndarray)
            assert len(datarow["samples"]) == nshots
            assert isinstance(datarow["depth"], int)
            assert datarow["depth"] == depths[count % len(depths)]
            if not datarow["countX"]:
                assert np.array_equal(
                    datarow["samples"], np.zeros(datarow["samples"].shape)
                )
            else:
                theor_outcome = datarow["countX"] % 2
                assert not np.array_equal(
                    datarow["samples"],
                    np.zeros(datarow["samples"].shape) + theor_outcome,
                )
        assert isinstance(myfaultyexperiment.dataframe, pd.DataFrame)


@pytest.mark.parametrize("nqubits", [1, 4])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("qubits", [[0], [2]])
def test_post_processing(
    nqubits: int, depths: list, runs: int, nshots: int, qubits: list
):
    if max(qubits) > nqubits - 1:
        pass
    else:
        # Build the noise model.
        noise = noisemodels.PauliErrorOnX()
        # Test exectue an experiment.
        myfactory1 = XIdrb.ModuleFactory(nqubits, list(depths) * runs, qubits)
        myfaultyexperiment = XIdrb.ModuleExperiment(
            myfactory1, nshots=nshots, noise_model=noise
        )
        myfaultyexperiment.perform(myfaultyexperiment.execute)
        XIdrb.post_processing_sequential(myfaultyexperiment)
        aggr_df = XIdrb.get_aggregational_data(myfaultyexperiment)
        assert len(aggr_df) == 1 and aggr_df.index[0] == "filter"
        assert "depth" in aggr_df.columns
        assert "data" in aggr_df.columns
        assert "2sigma" in aggr_df.columns
        assert "fit_func" in aggr_df.columns
        assert "popt" in aggr_df.columns
        assert "perr" in aggr_df.columns


@pytest.mark.parametrize("nqubits", [1, 5])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("qubits", [[0], [2]])
def test_build_report(depths: list, nshots: int, nqubits: int, runs: int, qubits: list):
    if max(qubits) > nqubits - 1:
        pass
    else:
        noise_params = [0.01, 0.1, 0.05]
        # Build the noise model.
        noise = noisemodels.PauliErrorOnX(*noise_params)
        # Test exectue an experiment.
        myfactory1 = XIdrb.ModuleFactory(nqubits, depths * runs, qubits)
        myfaultyexperiment = XIdrb.ModuleExperiment(
            myfactory1, nshots=nshots, noise_model=noise
        )
        myfaultyexperiment.perform(myfaultyexperiment.execute)
        XIdrb.post_processing_sequential(myfaultyexperiment)
        aggr_df = XIdrb.get_aggregational_data(myfaultyexperiment)
        report_figure, _ = XIdrb.build_report(myfaultyexperiment, aggr_df)
        assert isinstance(report_figure, Figure)
