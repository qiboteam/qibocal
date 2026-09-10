"""Tests for TWPA frequency and offset sweeper protocol."""

import numpy as np
import pytest

from qibocal.protocols import PROTOCOLS
from qibocal.protocols.twpa.frequency_offset import (
    TwpaFrequencyOffsetData,
    TwpaFrequencyOffsetParameters,
    TwpaFrequencyOffsetResults,
    _acquisition,
    _fit,
    _plot,
    twpa_frequency_offset,
    twpa_sweep,
)
from qibocal.protocols.utils import to_range


def test_twpa_protocols_registration():
    assert twpa_frequency_offset is twpa_sweep
    assert "twpa_frequency_offset" in PROTOCOLS
    assert "twpa_sweep" in PROTOCOLS


def test_parameters_validation():
    # Valid parameters with RangeLike formats and probes list
    params = TwpaFrequencyOffsetParameters(
        amplitude=[0.0, 0.5, 0.1],
        frequency=["center", 20_000_000, 2_000_000],
        probes=[7_000_000_000, 7_200_000_000],
    )
    assert to_range(params.amplitude) == (0.0, 0.5, 0.1)
    assert params.probes == [7_000_000_000, 7_200_000_000]

    params_rangelike = TwpaFrequencyOffsetParameters(
        amplitude=("linspace", 0.0, 0.4, 5),
        frequency=(6_000_000_000, 6_100_000_000, 2_000_000),
    )
    assert params_rangelike.amplitude == ("linspace", 0.0, 0.4, 5)
    assert params_rangelike.probes is None


def test_acquisition_invalid_amplitude(platform):
    # Invalid amplitude (offset) >= 1
    params_invalid_offset = TwpaFrequencyOffsetParameters(
        amplitude=[0.0, 1.5, 0.5],
        frequency=["center", 20_000_000, 2_000_000],
        probes=[7_000_000_000],
    )
    with pytest.raises(
        ValueError, match="TWPA amplitude values must be between -1 and 1"
    ):
        _acquisition(params_invalid_offset, platform, [0])


def test_acquisition_and_fit(platform, tmp_path):
    targets = [0, 1]
    probes = [7_000_000_000, 7_200_000_000]
    params = TwpaFrequencyOffsetParameters(
        amplitude=[0.1, 0.4, 0.1],
        frequency=["center", 10_000_000, 2_000_000],
        probes=probes,
        nshots=100,
    )

    data = _acquisition(params, platform, targets)
    assert isinstance(data, TwpaFrequencyOffsetData)
    assert data.probes == probes

    for qubit in targets:
        assert qubit in data.data
        assert qubit in data.offset
        assert qubit in data.frequency
        assert qubit in data.amplitude
        assert qubit in data.frequency
        assert qubit in data.reference_value

        # Shape check: (N_amplitude, N_twpa_freq, N_probes, 2)
        n_offset = len(data.offset[qubit])
        n_twpa_freq = len(data.frequency[qubit])
        assert data.data[qubit].shape[0] == n_offset
        assert data.data[qubit].shape[1] == n_twpa_freq
        assert data.data[qubit].shape[2] == len(probes)
        assert data.data[qubit].shape[3] == 2

        ref_arr = data.reference_value_array(qubit)
        assert ref_arr.shape == (len(probes), 2)

    # Test Fit
    fit_res = _fit(data)
    assert isinstance(fit_res, TwpaFrequencyOffsetResults)
    for qubit in targets:
        assert qubit in fit_res.frequency
        assert qubit in fit_res.offset
        assert qubit in fit_res.frequency
        assert qubit in fit_res.amplitude
        assert fit_res.frequency[qubit] in data.frequency[qubit]
        assert fit_res.offset[qubit] in data.offset[qubit]
        assert fit_res.data[qubit].shape == (n_offset, n_twpa_freq)

    # Test Plot with and without fit
    figs, report = _plot(data, fit_res, targets[0])
    assert len(figs) == 1
    assert "TWPA Frequency [Hz]" in report
    assert "TWPA Amplitude" in report

    figs_no_fit, report_no_fit = _plot(data, None, targets[0])
    assert len(figs_no_fit) == 1
    assert report_no_fit == ""

    # Test serialization
    data.save(tmp_path)
    loaded_data = TwpaFrequencyOffsetData.load(tmp_path)
    for qubit in targets:
        np.testing.assert_array_equal(data.data[qubit], loaded_data.data[qubit])
        assert data.offset[qubit] == loaded_data.offset[qubit]
        assert data.frequency[qubit] == loaded_data.frequency[qubit]
        assert data.probes == loaded_data.probes


def test_acquisition_default_probes(platform):
    targets = [0]
    params = TwpaFrequencyOffsetParameters(
        amplitude=[0.1, 0.3, 0.1],
        frequency=["center", 10_000_000, 5_000_000],
        nshots=50,
    )
    data = _acquisition(params, platform, targets)
    assert len(data.probes) == 1
    assert data.data[targets[0]].shape[2] == 1
