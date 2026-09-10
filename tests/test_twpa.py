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
    # Valid parameters with RangeLike formats
    params = TwpaFrequencyOffsetParameters(
        amplitude=[0.0, 0.5, 0.1],
        frequency=["center", 20_000_000, 2_000_000],
        probe_frequency=["center", 10_000_000, 1_000_000],
    )
    assert to_range(params.amplitude) == (0.0, 0.5, 0.1)

    params_rangelike = TwpaFrequencyOffsetParameters(
        amplitude=("linspace", 0.0, 0.4, 5),
        frequency=(6_000_000_000, 6_100_000_000, 2_000_000),
        probe_frequency=(7_000_000_000, 7_010_000_000, 1_000_000),
    )
    assert params_rangelike.amplitude == ("linspace", 0.0, 0.4, 5)


def test_acquisition_invalid_amplitude(platform):
    # Invalid amplitude (offset) >= 1
    params_invalid_offset = TwpaFrequencyOffsetParameters(
        amplitude=[0.0, 1.5, 0.5],
        frequency=["center", 20_000_000, 2_000_000],
        probe_frequency=["center", 10_000_000, 1_000_000],
    )
    with pytest.raises(
        ValueError, match="TWPA amplitude values must be between -1 and 1"
    ):
        _acquisition(params_invalid_offset, platform, [0])


def test_acquisition_and_fit(platform, tmp_path):
    targets = [0, 1]
    params = TwpaFrequencyOffsetParameters(
        amplitude=[0.1, 0.4, 0.1],
        frequency=["center", 10_000_000, 2_000_000],
        probe_frequency=["center", 10_000_000, 2_000_000],
        nshots=100,
    )

    data = _acquisition(params, platform, targets)
    assert isinstance(data, TwpaFrequencyOffsetData)

    for qubit in targets:
        assert qubit in data.data
        assert qubit in data.offset
        assert qubit in data.frequency
        assert qubit in data.amplitude
        assert qubit in data.frequency
        assert qubit in data.probes

        # Shape check: (N_offset, N_twpa_freq, N_ro_freq, 2)
        n_offset = len(data.offset[qubit])
        n_twpa_freq = len(data.frequency[qubit])
        assert data.data[qubit].shape[0] == n_offset
        assert data.data[qubit].shape[1] == n_twpa_freq
        assert data.data[qubit].shape[3] == 2

        ref_arr = data.reference_value_array(qubit)
        assert ref_arr.shape[1] == 2
        assert ref_arr.shape[0] == data.data[qubit].shape[2]

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
