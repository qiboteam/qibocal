import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.signal import lfilter

from qibocal.protocols import cryoscope
from qibocal.protocols.flux_dependence.cryoscope import (
    CryoscopeData,
    CryoscopeResults,
)

TEST_FILE_DIR = Path(__file__).resolve().parent


def test_cryoscope_acquisition(platform):
    target = [0]

    params = cryoscope.parameters_type.load(
        dict(
            duration_min=1,
            duration_max=10,
            duration_step=1,
            flux_pulse_amplitude=0.1,
        )
    )

    cryoscope_data, _ = cryoscope.acquisition(params, platform, target)
    assert isinstance(cryoscope_data, CryoscopeData)


def test_cryoscope_postprocessing():
    datafolder = TEST_FILE_DIR / "cryoscope_data" / "data" / "cryoscope 1-0"
    metadatafolder = TEST_FILE_DIR / "cryoscope_data" / "meta.json"

    data = cryoscope.data_type.load(datafolder)
    fit_results, _ = cryoscope.fit(data)
    assert isinstance(fit_results, CryoscopeResults)

    results = cryoscope.results_type.load(datafolder)

    with open(metadatafolder) as file:
        metadata = json.load(file)

    targets = metadata["targets"]

    for target in targets:
        assert math.isclose(fit_results.tau[target], results.tau[target], rel_tol=1e-4)
        assert math.isclose(
            fit_results.exp_amplitude[target],
            results.exp_amplitude[target],
            rel_tol=1e-4,
        )

        # check IIR
        assert np.allclose(
            fit_results.feedforward_taps_iir[target],
            results.feedforward_taps_iir[target],
            rtol=1e-4,
        )

        # first check on FIR
        assert np.allclose(
            fit_results.feedback_taps[target],
            results.feedback_taps[target],
            rtol=1e-4,
        )

    # control that corrections are actually working
    for target in targets:
        all_corrections = lfilter(
            fit_results.feedforward_taps[target],
            results.feedback_taps[target],
            fit_results.step_response[target],
        )
        all_corrections = all_corrections[: len(fit_results.feedforward_taps[target])]
        assert np.ptp(all_corrections) <= 1e-2


def test_cryoscope_plot():
    datafolder = TEST_FILE_DIR / "cryoscope_data" / "data" / "cryoscope 1-0"
    metadatafolder = TEST_FILE_DIR / "cryoscope_data" / "meta.json"

    results = cryoscope.results_type.load(datafolder)
    data = cryoscope.data_type.load(datafolder)

    with open(metadatafolder) as file:
        metadata = json.load(file)

    targets = metadata["targets"]

    for target in targets:
        figs, fitting_report = cryoscope.report(data, results, target)

        assert isinstance(figs, list)
        assert all(isinstance(fig, go.Figure) for fig in figs)
        assert isinstance(fitting_report, str)

        for fig in figs:
            assert len(fig.data) == 5
            assert fig.data[0].name == "X"
            assert fig.data[1].name == "Y"
            assert fig.data[2].name == "Uncorrected waveform"
            assert fig.data[3].name == "IIR corrections"
            assert fig.data[4].name == "FIR + IIR corrections"
