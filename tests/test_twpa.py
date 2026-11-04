"""Tests for TWPA frequency and offset sweeper protocol."""

from types import SimpleNamespace

import numpy as np
import pytest
from qibolab import IqChannel, IqConfig, OscillatorConfig

from qibocal.protocols import PROTOCOLS
from qibocal.protocols.twpa.frequency_offset import (
    PUMP_OFF_POWER,
    TwpaFrequencyOffsetData,
    TwpaFrequencyOffsetParameters,
    TwpaFrequencyOffsetResults,
    _acquisition,
    _fit,
    _plot,
    _pump_power,
    _select,
    _update,
    twpa_frequency_offset,
    twpa_sweep,
)
from qibocal.protocols.utils import readout_frequency, to_range


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

    params_loaded = TwpaFrequencyOffsetParameters.load(
        {
            "amplitude": [0.0, 0.5, 0.1],
            "frequency": ["center", 20_000_000, 2_000_000],
            "probes": [7_000_000_000, 7_200_000_000],
            "nshots": 100,
        }
    )
    assert to_range(params_loaded.amplitude) == (0.0, 0.5, 0.1)
    assert params_loaded.probes == [7_000_000_000, 7_200_000_000]
    assert params_loaded.nshots == 100


@pytest.mark.parametrize("amplitude", [[0.0, 2.0, 0.5], [-1.5, 0.0, 0.5]])
def test_acquisition_invalid_amplitude(platform, amplitude):
    params_invalid_offset = TwpaFrequencyOffsetParameters.load(
        {
            "amplitude": amplitude,
            "frequency": ["center", 20_000_000, 2_000_000],
            "probes": [7_000_000_000],
        }
    )
    with pytest.raises(
        ValueError, match="TWPA amplitude values must be between -1 and 1"
    ):
        _acquisition(params_invalid_offset, platform, [0])


def test_acquisition_amplitude_boundaries(platform):
    params = TwpaFrequencyOffsetParameters.load(
        {
            "amplitude": [-1.0, 1.5, 1.0],
            "frequency": ["center", 10_000_000, 5_000_000],
            "nshots": 50,
        }
    )
    data = _acquisition(params, platform, [0])
    assert data.offset[0] == [-1.0, 0.0, 1.0]


@pytest.mark.parametrize("reference", [0.0, 1e-12, np.nan, np.inf])
def test_invalid_reference(reference):
    data = TwpaFrequencyOffsetData(
        data={0: np.ones((1, 1, 1, 2))},
        reference_value={0: [[reference, 0.0]]},
        frequency={0: [6_000_000_000]},
        offset={0: [0.5]},
    )
    with np.errstate(divide="raise", invalid="raise"):
        for operation in (lambda: _fit(data), lambda: _plot(data, None, 0)):
            with pytest.raises(
                ValueError, match="TWPA-off reference magnitude for qubit 0"
            ):
                operation()


@pytest.mark.parametrize("reference", [1.0, 1e-6])
def test_averaged_gain(reference):
    data = TwpaFrequencyOffsetData(
        data={0: reference * np.array([[[[3.0, 4.0], [0.0, 3.0]]]])},
        reference_value={0: [[reference, 0.0], [0.0, 3 * reference]]},
    )
    np.testing.assert_allclose(data.averaged_gain(0), [[20 * np.log10(2)]])


def test_acquisition_and_fit(platform, tmp_path):
    targets = [0, 1]
    probes = [7_000_000_000, 7_200_000_000]
    params = TwpaFrequencyOffsetParameters.load(
        {
            "amplitude": [0.1, 0.4, 0.1],
            "frequency": ["center", 10_000_000, 2_000_000],
            "probes": probes,
            "nshots": 100,
        }
    )

    data = _acquisition(params, platform, targets)
    assert isinstance(data, TwpaFrequencyOffsetData)
    assert data.probes == probes

    for qubit in targets:
        assert qubit in data.data
        assert qubit in data.offset
        assert qubit in data.frequency
        assert qubit in data.reference_value
        assert qubit in data.power
        assert data.power[qubit] == 10.0

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
        assert qubit in fit_res.gain
        assert fit_res.frequency[qubit] in data.frequency[qubit]
        assert fit_res.offset[qubit] in data.offset[qubit]
        assert fit_res.power[qubit] == pytest.approx(
            10.0 + 20 * np.log10(abs(fit_res.offset[qubit]))
        )

    # Test Plot with and without fit
    figs, report = _plot(data, fit_res, targets[0])
    assert len(figs) == 1
    fig = figs[0]
    assert fig.data[0].colorbar.title.text == "TWPA Gain [dB]"
    assert fig.layout.yaxis2 is not None
    assert fig.layout.yaxis2.title.text == "Pump power [dB]"
    assert fig.layout.yaxis2.ticktext == tuple(
        f"{np.round(data.power[targets[0]] + 20 * np.log10(abs(offset)), 1)}"
        for offset in data.offset[targets[0]]
    )
    assert fig.layout.yaxis2.side == "right"
    assert fig.layout.yaxis2.overlaying == "y"
    assert fig.layout.showlegend is True
    assert fig.layout.legend.orientation == "h"
    assert any(trace.yaxis == "y2" for trace in fig.data)
    # mock data are noise: the point may lie at the edge of the small scan
    point = (
        "Candidate Point (scan edge)" if fit_res.is_candidate(0) else "Optimal Point"
    )
    assert any(
        getattr(trace, "name", None) == point and trace.showlegend is True
        for trace in fig.data
    )
    if fit_res.is_candidate(0):
        assert "Fit failed" in report
    else:
        assert "Pump Frequency [Hz]" in report
        assert "Pump Amplitude" in report
        assert "Pump power [dB]" in report
        assert "TWPA Gain [dB]" in report

    figs_no_fit, report_no_fit = _plot(data, None, targets[0])
    assert len(figs_no_fit) == 1
    assert figs_no_fit[0].data[0].colorbar.title.text == "TWPA Gain [dB]"
    assert figs_no_fit[0].layout.yaxis2.title.text == "Pump power [dB]"
    assert any(trace.yaxis == "y2" for trace in figs_no_fit[0].data)
    assert report_no_fit == ""

    # Test serialization
    data.save(tmp_path)
    loaded_data = TwpaFrequencyOffsetData.load(tmp_path)
    for qubit in targets:
        np.testing.assert_array_equal(data.data[qubit], loaded_data.data[qubit])
        assert data.offset[qubit] == loaded_data.offset[qubit]
        assert data.frequency[qubit] == loaded_data.frequency[qubit]
        assert data.probes == loaded_data.probes
        assert data.power == loaded_data.power

    fit_res.save(tmp_path)
    loaded_fit = TwpaFrequencyOffsetResults.load(tmp_path)
    for qubit in targets:
        assert fit_res.gain[qubit] == loaded_fit.gain[qubit]
        assert fit_res.frequency[qubit] == loaded_fit.frequency[qubit]
        assert fit_res.offset[qubit] == loaded_fit.offset[qubit]


def test_acquisition_default_probes(platform, mocker):
    targets = [0, 1]
    params = TwpaFrequencyOffsetParameters.load(
        {
            "amplitude": [0.1, 0.3, 0.1],
            "frequency": ["center", 10_000_000, 5_000_000],
            "nshots": 50,
        }
    )
    spy = mocker.spy(platform, "execute")
    data = _acquisition(params, platform, targets)
    assert data.probes is None
    # Exactly 2 platform.execute calls: 1 for reference scan, 1 for 2D sweep
    assert spy.call_count == 2
    sweep_call = spy.call_args_list[1]
    updates = sweep_call.kwargs["updates"][0]
    for q in targets:
        probe_ch = platform.qubits[q].probe
        ro_freq = readout_frequency(q, platform)
        assert updates[probe_ch]["frequency"] == ro_freq
        assert data.data[q].shape[2] == 1
        assert data.reference_value_array(q).shape == (1, 2)

    sweepers = sweep_call.args[1]
    assert len(sweepers) == 2
    assert all(sw.range is not None for sw in sweepers[0])
    assert all(sw.range is not None for sw in sweepers[1])


def test_reference_pump_off(platform, mocker):
    params = TwpaFrequencyOffsetParameters.load(
        {
            "amplitude": [0.1, 0.3, 0.1],
            "frequency": ["center", 10_000_000, 5_000_000],
            "nshots": 50,
        }
    )
    spy = mocker.spy(platform, "execute")
    _acquisition(params, platform, [0])
    reference_call = spy.call_args_list[0]
    assert len(reference_call.args) == 1
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    probe_updates, off_updates = reference_call.kwargs["updates"]
    assert all("offset" not in up for up in probe_updates.values())
    assert off_updates == {pump: {"power": PUMP_OFF_POWER}}


def _pump_platform(channels: dict, configs: dict) -> SimpleNamespace:
    return SimpleNamespace(channels=channels, config=configs.__getitem__)


def test_pump_power():
    oscillator = _pump_platform({}, {"pump": OscillatorConfig(frequency=6e9, power=-5)})
    assert _pump_power(oscillator, "pump") == -5

    channel = _pump_platform(
        {"pump": IqChannel(path="8/o2", lo="pump/lo")},
        {
            "pump": IqConfig(frequency=6e9),
            "pump/lo": OscillatorConfig(frequency=6e9, power=-21.4),
        },
    )
    assert _pump_power(channel, "pump") == -21.4


@pytest.mark.parametrize(
    "channels, match",
    [({}, "not found"), ({"pump": IqChannel(path="8/o2")}, "has no LO")],
)
def test_pump_power_invalid(channels, match):
    platform = _pump_platform(channels, {"pump": IqConfig(frequency=6e9)})
    with pytest.raises(ValueError, match=match):
        _pump_power(platform, "pump")


def test_update_oscillator_pump(platform):
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    results = TwpaFrequencyOffsetResults(
        frequency={0: 6.1e9}, offset={0: 0.5}, gain={0: 20.0}, power={0: -12.0}
    )
    _update(results, platform, 0)
    assert platform.config(pump) == OscillatorConfig(frequency=6.1e9, power=-12.0)


def test_update_iq_pump():
    updates = {}
    platform = SimpleNamespace(
        channels={
            "acq": SimpleNamespace(twpa_pump="pump"),
            "pump": IqChannel(path="8/o2", lo="pump/lo"),
        },
        qubits={0: SimpleNamespace(acquisition="acq")},
        config={"pump": IqConfig(frequency=6e9)}.__getitem__,
        update=updates.update,
    )
    results = TwpaFrequencyOffsetResults(
        frequency={0: 6.1e9}, offset={0: 0.5}, gain={0: 20.0}, power={0: -27.4}
    )
    _update(results, platform, 0)
    assert updates == {
        "configs.pump.frequency": 6.1e9,
        "configs.pump/lo.power": -27.4,
    }


def landscape(offsets, frequencies, cliff=0.8, center=6.0e9):
    """Gain growing with the pump power up to an over-pumping cliff at `cliff`.

    The gain also decreases away from a central pump frequency.
    """
    offsets, frequencies = np.asarray(offsets), np.asarray(frequencies)
    power = 20 * np.log10(np.abs(offsets))[:, np.newaxis]
    detuning = ((frequencies - center) / 20e6)[np.newaxis, :] ** 2
    gain = 20 + 1.5 * power - 2 * detuning
    return np.where(np.abs(offsets)[:, np.newaxis] > cliff, -5.0, gain)


def landscape_data(offsets, frequencies, **kwargs):
    gain = landscape(offsets, frequencies)
    magnitude = 10 ** (gain / 20)
    return TwpaFrequencyOffsetData(
        data={0: np.stack([magnitude, np.zeros_like(magnitude)], axis=-1)[:, :, None]},
        reference_value={0: [[1.0, 0.0]]},
        offset={0: list(offsets)},
        frequency={0: list(frequencies)},
        power={0: 0.0},
        **kwargs,
    )


OFFSETS = np.round(np.arange(0.1, 1.0, 0.05), 2)
FREQUENCIES = 6.0e9 + np.arange(-30e6, 31e6, 5e6)


def test_select_away_from_overpumping():
    gain = landscape(OFFSETS, FREQUENCIES)
    plain = _select(gain, OFFSETS, 0, 0.0)
    robust = _select(gain, OFFSETS, 1, 0.0)

    # the plain maximum is just below the cliff, the robust one a step further
    assert OFFSETS[plain[0]] == 0.8
    assert OFFSETS[robust[0]] == 0.75
    assert FREQUENCIES[robust[1]] == 6.0e9
    assert not plain[2] and not robust[2]


def test_select_backoff():
    gain = landscape(OFFSETS, FREQUENCIES)
    i, _, candidate = _select(gain, OFFSETS, 1, 2.0)

    # about 2 dB below 0.75, i.e. 0.6
    assert OFFSETS[i] == 0.6
    assert not candidate


def test_candidate_at_scan_edge(platform):
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    before = platform.config(pump)
    # the scan stops before the cliff: the optimum is at the largest amplitude
    data = landscape_data(OFFSETS[OFFSETS < 0.6], FREQUENCIES, power_backoff=2.0)
    fit = _fit(data)

    assert 0 in fit
    assert fit.is_candidate(0)
    assert fit.offset[0] == 0.55
    figures, report = _plot(data, fit, 0)
    assert "Fit failed" in report
    assert "pump amplitude range" in report
    assert "Pump Amplitude" not in report
    names = [trace.name for trace in figures[0].data]
    assert "Candidate Point (scan edge)" in names
    assert "Optimal Point" not in names
    _update(fit, platform, 0)
    assert platform.config(pump) == before


def test_results_without_candidate(platform):
    """Results of previous versions, without candidates, are still applied."""
    results = TwpaFrequencyOffsetResults(
        frequency={0: 6.1e9}, offset={0: 0.5}, gain={0: 20.0}, power={0: -12.0}
    )
    assert 0 in results
    assert not results.is_candidate(0)
    _update(results, platform, 0)
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    assert platform.config(pump).frequency == 6.1e9
