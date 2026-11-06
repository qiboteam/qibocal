"""Tests for the coupler JAZZ protocol."""

import numpy as np
import pytest

from qibocal.calibration.calibration import CouplerCalibration
from qibocal.protocols import PROTOCOLS
from qibocal.protocols.couplers.jazz import (
    CouplerJAZZParameters,
    _acquisition,
    _fit,
    _plot,
    _update,
    coupler_jazz,
    find_nulling_point,
)
from qibocal.protocols.utils import to_range

PAIR = (0, 1)
COUPLER = 1

PARAMETERS = {
    "delays_settings": [0, 400, 20],
    "coupler_offset_range": [-0.1, 0.1, 0.05],
    "nshots": 100,
}


@pytest.fixture
def coupler_platform(platform):
    """Mock platform with the coupler linked to the qubit pair."""
    platform.calibration.two_qubits[PAIR].coupler = COUPLER
    platform.calibration.couplers[COUPLER] = CouplerCalibration()
    return platform


def test_coupler_jazz_registration():
    assert PROTOCOLS["coupler_jazz"] is coupler_jazz


def test_coupler_jazz(coupler_platform):
    params = CouplerJAZZParameters.load(PARAMETERS)
    data = _acquisition(params, coupler_platform, [PAIR])

    offsets = np.arange(*to_range(PARAMETERS["coupler_offset_range"]))
    np.testing.assert_allclose(data.pair_offsets(PAIR), offsets)
    assert len(data[PAIR]) == len(offsets) * len(data.delays)

    results = _fit(data)
    null_bias, null_zz = results.nulling_point[PAIR]
    assert null_bias in data.pair_offsets(PAIR)

    figures, report = _plot(data, PAIR, results)
    assert len(figures) == 1
    assert report

    _update(results, coupler_platform, PAIR)
    coupler_flux = coupler_platform.couplers[COUPLER].flux
    assert coupler_platform.config(coupler_flux).offset == null_bias
    assert coupler_platform.calibration.couplers[COUPLER].idle_bias == null_bias
    assert coupler_platform.calibration.two_qubits[PAIR].coupling == [null_zz, 0.0]


def test_coupler_jazz_without_coupler(platform):
    params = CouplerJAZZParameters.load(PARAMETERS)
    with pytest.raises(ValueError, match="Coupler ID"):
        _acquisition(params, platform, [PAIR])


def test_find_nulling_point():
    # [offset, amplitude, decay] per coupler offset
    popts = {
        -0.1: [0.9, 0.4, 3.0],
        0.0: [0.5, 0.0, 2.0],
        0.1: [0.9, 0.4, 0.0],
    }
    offsets = np.array(list(popts))
    # only the decay term: the slowest signal is selected
    assert find_nulling_point(popts, offsets, weight=1.0) == 2
    # only the signal term: the flat signal centered at 1/2 is selected
    assert find_nulling_point(popts, offsets, weight=0.0) == 1
