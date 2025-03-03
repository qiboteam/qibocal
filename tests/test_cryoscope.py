import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from qibocal.protocols.two_qubit_interaction import cryoscope

TEST_FILE_DIR = Path(__file__).resolve().parent
DATAFOLDER = TEST_FILE_DIR / "cryoscope_data" / "data" / "cryoscope-0"
METADATAFOLDER = DATAFOLDER.parent.parent / "meta.json"
SAMPLING_RATE = 1
# Instrument sampling rate in GSamples


def test_cryoscope_postprocessing():
    data = cryoscope.data_type.load(DATAFOLDER)
    fit_results = cryoscope.fit(data)

    results = cryoscope.results_type.load(DATAFOLDER)

    with open(METADATAFOLDER) as file:
        metadata = json.load(file)

    targets = metadata["targets"]

    for target in targets:
        assert math.isclose(
            fit_results[0].tau[target], results.tau[target], rel_tol=1e-4
        )
        assert math.isclose(
            fit_results[0].exp_amplitude[target],
            results.exp_amplitude[target],
            rel_tol=1e-4,
        )

        # check IIR
        assert np.allclose(
            fit_results[0].feedforward_taps_iir[target],
            results.feedforward_taps_iir[target],
            rtol=1e-4,
        )

        # first check on FIR
        assert np.allclose(
            fit_results[0].feedback_taps[target],
            results.feedback_taps[target],
            rtol=1e-4,
        )

    # control that corrections are actually working
    for target in targets:
        all_corrections = lfilter(
            fit_results[0].feedforward_taps[target],
            results.feedback_taps[target],
            fit_results[0].step_response[target],
        )
        all_corrections = all_corrections[
            : len(fit_results[0].feedforward_taps[target])
        ]
        assert np.ptp(all_corrections) <= 1e-2
