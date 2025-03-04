import json
import math
from pathlib import Path

import numpy as np
from qibolab import create_dummy
from scipy.signal import lfilter

from qibocal.protocols.two_qubit_interaction import cryoscope
from qibocal.protocols.two_qubit_interaction.cryoscope import (
    CryoscopeData,
    CryoscopeParameters,
    CryoscopeResults,
)

TEST_FILE_DIR = Path(__file__).resolve().parent
SAMPLING_RATE = 1

# Instrument sampling rate in GSamples


def test_acquisition():

    platform = create_dummy()
    target = [0]
    duration_min = 1
    duration_max = 10
    duration_step = 1
    flux_pulse_amplitude = 0.1

    params = CryoscopeParameters(
        duration_min, duration_max, duration_step, flux_pulse_amplitude
    )
    # pdb.set_trace()

    cryoscope_data = cryoscope.acquisition(params, platform, target)
    assert type(cryoscope_data) == CryoscopeData


def test_cryoscope_postprocessing():

    datafolder = TEST_FILE_DIR / "cryoscope_data" / "data" / "cryoscope-0"
    metadatafolder = TEST_FILE_DIR / "cryoscope_data" / "meta.json"

    data = cryoscope.data_type.load(datafolder)
    fit_results, _ = cryoscope.fit(data)
    assert type(fit_results) == CryoscopeResults

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


def main():
    test_acquisition()
    test_cryoscope_postprocessing()


if __name__ == "__main__":
    main()
