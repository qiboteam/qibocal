from pathlib import Path

import pytest

from qibocal.protocols import cryoscope
from qibocal.protocols.flux_dependence.cryoscope import (
    CryoscopeData,
)

TEST_FILE_DIR = Path(__file__).resolve().parent


def test_cryoscope_acquisition(platform):
    target = [0]

    params = cryoscope.parameters_type.load(
        {
            "duration_min": 1,
            "duration_max": 10,
            "duration_step": 1,
            "flux_pulse_amplitude": 0.1,
        }
    )

    cryoscope_data, _ = cryoscope.acquisition(params, platform, target)
    assert isinstance(cryoscope_data, CryoscopeData)


def test_cryoscope_acquisition_raises_without_flux_coefficients(platform):
    target = 0
    platform.calibration.single_qubits[target].qubit.flux_coefficients = None

    params = cryoscope.parameters_type.load(
        {
            "duration_min": 1,
            "duration_max": 10,
            "duration_step": 1,
            "flux_pulse_amplitude": 0.1,
        }
    )

    with pytest.raises(
        ValueError, match="Cannot run cryoscope without flux coefficients"
    ):
        cryoscope.acquisition(params, platform, [target])
