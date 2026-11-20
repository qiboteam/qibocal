"""Tests for TWPA calibration and frequency and offset sweeper protocols."""

from types import SimpleNamespace

import numpy as np
import pytest
from qibolab import (
    Acquisition,
    IqChannel,
    IqConfig,
    OscillatorConfig,
    Parameter,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    Sweeper,
)

from qibocal.protocols import PROTOCOLS
from qibocal.protocols.twpa.frequency_offset import (
    TwpaFrequencyOffsetData,
    TwpaFrequencyOffsetFitParameters,
    TwpaFrequencyOffsetParameters,
    TwpaFrequencyOffsetResults,
    _acquisition,
    _fit,
    _plot,
    twpa_frequency_offset,
    twpa_sweep,
)
from qibocal.protocols.twpa.twpa import (
    TwpaCalibrationParameters,
    _reference_scan,
    _sweep,
    _twpa_scan,
)
from qibocal.protocols.twpa.twpa import (
    _fit as calibration_fit,
)
from qibocal.protocols.twpa.twpa import (
    _plot as calibration_plot,
)
from qibocal.protocols.twpa.utils import PUMP_OFF_POWER, Pumps, probe_labels, pump_point
from qibocal.protocols.twpa.utils import pump_power as _pump_power
from qibocal.protocols.twpa.utils import select as _select
from qibocal.protocols.twpa.utils import update_point as _update
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

        # each sweep is bracketed by a reference before and after it
        assert data.references(qubit).shape == (len(probes), 2, 2)

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
    # minimum and average maps, and per-probe maps, since there are two probes
    assert len(figs) == 3
    fig = figs[0]
    assert fig.data[0].colorbar.title.text == "Minimum TWPA Gain across probes [dB]"
    assert figs[1].data[0].colorbar.title.text == "Average TWPA Gain [dB]"
    # the targets share the pump: the probes of both are aggregated
    assert data.group(targets[0]) == targets
    assert fit_res.frequency[0] == fit_res.frequency[1]
    assert fit_res.offset[0] == fit_res.offset[1]
    labels = [f"Qubit {q} Probe {p} GHz" for q in targets for p in ("7.0000", "7.2000")]
    assert [t.text for t in figs[2].layout.annotations] == labels
    assert _plot(data, fit_res, targets[1])[0] == []
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
        assert "Minimum TWPA Gain across probes [dB]" in report
        for label in labels:
            assert f"{label} Gain [dB]" in report

    figs_no_fit, report_no_fit = _plot(data, None, targets[0])
    assert len(figs_no_fit) == 3
    assert (
        figs_no_fit[0].data[0].colorbar.title.text
        == "Minimum TWPA Gain across probes [dB]"
    )
    assert figs_no_fit[0].layout.yaxis2.title.text == "Pump power [dB]"
    assert any(trace.yaxis == "y2" for trace in figs_no_fit[0].data)
    # no fit results, but the aggregation across the shared pump is explained
    assert "shared by qubits 0, 1" in report_no_fit
    assert "<table" not in report_no_fit

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
    # Exactly 3 platform.execute calls: the 2D sweep, bracketed by two references
    assert spy.call_count == 3
    assert [len(call.args) for call in spy.call_args_list] == [1, 2, 1]
    sweep_call = spy.call_args_list[1]
    updates = sweep_call.kwargs["updates"][0]
    for q in targets:
        probe_ch = platform.qubits[q].probe
        ro_freq = readout_frequency(q, platform)
        assert updates[probe_ch]["frequency"] == ro_freq
        assert data.data[q].shape[2] == 1
        assert data.references(q).shape == (1, 2, 2)

    sweepers = sweep_call.args[1]
    assert len(sweepers) == 2
    assert all(sw.range is not None for sw in sweepers[0])
    assert all(sw.range is not None for sw in sweepers[1])


@pytest.mark.parametrize("pumps", [("a",), ("a", "a"), ("a", "b", "a")])
@pytest.mark.parametrize("legacy", [True, False])
def test_twpa_scan_pump_mapping(mocker, pumps, legacy):
    qubits = list(reversed(range(len(pumps))))
    configs = {
        "a": OscillatorConfig(frequency=6e9, power=0.1),
        "b": OscillatorConfig(frequency=7e9, power=0.2),
    }
    params = TwpaCalibrationParameters.load(
        {
            "nshots": 10,
            "relaxation_time": 0,
            "freq_width": 10e6,
            "freq_step": 1e6,
            "twpa_freq_width": 10e6,
            "twpa_freq_step": 2e6,
            "twpa_pow_width": 0.8,
            "twpa_pow_step": 0.1,
        }
    )
    if not legacy:
        params.probe_frequency = ("asym", (5e6, 5e6), 1e6)
        params.frequency = ("asym", (5e6, 5e6), 2e6)
        params.power = ("asym", (0.4, 0.4), 0.1)

    sequence = PulseSequence()
    sweepers = []
    channels = []
    results = {}
    for qubit in qubits:
        channel = f"{qubit}/acquisition"
        channels.append(channel)
        readout = Readout(
            acquisition=Acquisition(duration=100),
            probe=Pulse(duration=100, amplitude=0.1, envelope=Rectangular()),
        )
        sequence.append((channel, readout))
        sweeper = Sweeper(
            parameter=Parameter.frequency,
            range=params.probe_frequency_range(5e9),
            channels=[f"{qubit}/probe"],
        )
        sweepers.append(sweeper)
        results[readout.id] = np.ones((len(sweeper.values), 2))

    platform = mocker.Mock()
    platform.config.side_effect = configs.__getitem__
    platform.execute.return_value = results
    data = _twpa_scan(platform, sequence, sweepers, qubits, pumps, channels, params)

    assert (
        set(data.data)
        == set(data.twpa_frequency)
        == set(data.twpa_power)
        == set(qubits)
    )
    for qubit, pump, sweeper in zip(qubits, pumps, sweepers):
        expected_power = _sweep(params.power_range(configs[pump].power))
        expected_frequency = _sweep(params.frequency_range(configs[pump].frequency))
        np.testing.assert_array_equal(data.twpa_power[qubit], expected_power)
        np.testing.assert_array_equal(data.twpa_frequency[qubit], expected_frequency)
        assert data[qubit].shape == (
            len(expected_power),
            len(expected_frequency),
            len(sweeper.values),
            2,
        )
        data.reference_value[qubit] = results[
            list(sequence.channel(f"{qubit}/acquisition"))[-1].id
        ].tolist()

    unique_pumps = list(dict.fromkeys(pumps))
    expected_updates = [
        [
            {pump: {"power": power, "frequency": frequency}}
            for pump, power, frequency in zip(unique_pumps, powers, frequencies)
        ]
        for powers in zip(
            *(_sweep(params.power_range(configs[pump].power)) for pump in unique_pumps)
        )
        for frequencies in zip(
            *(
                _sweep(params.frequency_range(configs[pump].frequency))
                for pump in unique_pumps
            )
        )
    ]
    assert [
        call.kwargs["updates"] for call in platform.execute.call_args_list
    ] == expected_updates

    fit = calibration_fit(data)
    for qubit in qubits:
        assert fit.twpa_power[qubit] == data.twpa_power[qubit][0]
        assert fit.twpa_frequency[qubit] == data.twpa_frequency[qubit][0]
        assert len(calibration_plot(data, None, qubit)[0]) == 1
        assert len(calibration_plot(data, fit, qubit)[0]) == 1


def test_sweep_size_independent_of_center():
    """Ranges of equal width and step have the same size, unlike numpy.arange."""
    assert np.arange(3.0 - 0.4, 3.0 + 0.4, 0.1).size == 8
    assert np.arange(5.0 - 0.4, 5.0 + 0.4, 0.1).size == 9
    for center in (3.0, 5.0, 6.1e9):
        values = _sweep((center - 0.4, center + 0.4, 0.1))
        assert len(values) == 8
        assert values[0] == center - 0.4
    # otherwise, as numpy.arange
    for range_ in [
        (0.0, 1.0, 0.3),
        (0.0, 10.0, 3.0),
        (1.0, 0.0, -0.25),
        (0.0, 1.0, -0.1),
    ]:
        assert len(_sweep(range_)) == np.arange(*range_).size


@pytest.mark.parametrize(
    "missing, match",
    [
        ("freq_step", "'probe_frequency'"),
        ("twpa_freq_step", "'frequency'"),
        ("twpa_pow_step", "'power'"),
    ],
)
def test_missing_ranges(missing, match):
    parameters = {
        "freq_width": 10e6,
        "freq_step": 1e6,
        "twpa_freq_width": 10e6,
        "twpa_freq_step": 2e6,
        "twpa_pow_width": 0.8,
        "twpa_pow_step": 0.1,
    }
    del parameters[missing]
    params = TwpaCalibrationParameters.load(parameters)
    ranges = {
        "freq_step": params.probe_frequency_range,
        "twpa_freq_step": params.frequency_range,
        "twpa_pow_step": params.power_range,
    }
    with pytest.raises(ValueError, match=match):
        ranges[missing]()


def test_reference_scan_turns_pumps_back_on(mocker):
    twpa = mocker.Mock()
    platform = mocker.Mock()
    platform.execute.side_effect = RuntimeError("acquisition failed")

    with pytest.raises(RuntimeError, match="acquisition failed"):
        _reference_scan(platform, PulseSequence(), [], 10, 0, [], [twpa])
    twpa.device.off.assert_called_once_with()
    twpa.device.on.assert_called_once_with()


def test_shared_pump_switched_once(platform, mocker):
    from qibocal.protocols.twpa.twpa import _acquisition as calibration_acquisition

    pump = platform.instruments["twpa_pump"]
    platform.connect()
    off = mocker.spy(pump.device, "off")
    on = mocker.spy(pump.device, "on")
    params = TwpaCalibrationParameters.load(
        {
            "nshots": 10,
            "freq_width": 2e6,
            "freq_step": 1e6,
            "twpa_freq_width": 2e6,
            "twpa_freq_step": 1e6,
            "twpa_pow_width": 0.2,
            "twpa_pow_step": 0.1,
        }
    )
    execute = mocker.spy(platform, "execute")
    try:
        calibration_acquisition(params, platform, [0, 1])
        off.assert_called_once_with()
        on.assert_called_once_with()
    finally:
        platform.disconnect()

    # parallel probe sweeps, of the same size whatever their centers
    sweepers = execute.call_args_list[0].args[1][0]
    assert len({len(sweeper.values) for sweeper in sweepers}) == 1
    for sweeper, qubit in zip(sweepers, [0, 1]):
        center = readout_frequency(qubit, platform)
        assert sweeper.values.tolist() == _sweep(params.probe_frequency_range(center))


@pytest.mark.parametrize("override", [None, -40.0])
def test_reference_pump_off(platform, mocker, override):
    """The reference turns the pump off, at the default or overridden power."""
    params = TwpaFrequencyOffsetParameters.load(
        {
            "amplitude": [0.1, 0.3, 0.1],
            "frequency": ["center", 10_000_000, 5_000_000],
            "nshots": 50,
            **({} if override is None else {"off_power": override}),
        }
    )
    spy = mocker.spy(platform, "execute")
    _acquisition(params, platform, [0])
    reference_call = spy.call_args_list[0]
    assert len(reference_call.args) == 1
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    probe_updates, off_updates = reference_call.kwargs["updates"]
    assert all("offset" not in up for up in probe_updates.values())
    expected = PUMP_OFF_POWER if override is None else override
    assert off_updates == {pump: {"power": expected}}


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
    plain = _select(gain, 0)
    robust = _select(gain, 1)

    # the plain maximum is just below the cliff, the robust one a step further
    assert OFFSETS[plain[0]] == 0.8
    assert OFFSETS[robust[0]] == 0.75
    assert FREQUENCIES[robust[1]] == 6.0e9
    assert not plain[2] and not robust[2]


def test_neighbourhood_fit_parameter():
    """The neighbourhood is a fit parameter, not an acquisition one."""
    data = landscape_data(OFFSETS, FREQUENCIES)

    assert _fit(data).offset[0] == 0.75
    assert (
        _fit(data, TwpaFrequencyOffsetFitParameters(neighbourhood=0)).offset[0] == 0.8
    )
    assert twpa_frequency_offset.load_fit_parameters(
        {"neighbourhood": 0}
    ) == TwpaFrequencyOffsetFitParameters(neighbourhood=0)
    with pytest.raises(TypeError, match="neighbourhood"):
        TwpaFrequencyOffsetParameters.load(
            {
                "amplitude": [0.1, 0.3, 0.1],
                "frequency": [6e9, 6.1e9, 1e7],
                "neighbourhood": 0,
            }
        )


def test_candidate_at_scan_edge(platform):
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    before = platform.config(pump)
    # the scan stops before the cliff: the optimum is at the largest amplitude
    data = landscape_data(OFFSETS[OFFSETS < 0.6], FREQUENCIES)
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


def two_probes_data():
    """Two probes, the second one with a larger signal and a shifted gain peak."""
    gains = np.stack(
        [
            landscape(OFFSETS, FREQUENCIES),
            landscape(OFFSETS, FREQUENCIES, center=6.02e9) + 10,
        ],
        axis=2,
    )
    references = np.array([1.0, 10.0])
    magnitude = references * 10 ** (gains / 20)
    return gains, TwpaFrequencyOffsetData(
        data={0: np.stack([magnitude, np.zeros_like(magnitude)], axis=-1)},
        reference_value={0: [[ref, 0.0] for ref in references]},
        offset={0: list(OFFSETS)},
        frequency={0: list(FREQUENCIES)},
        power={0: 0.0},
        probes=[7.0e9, 7.2e9],
    )


def test_probe_gain():
    """Each probe gain is relative to its own reference, unlike the average."""
    gains, data = two_probes_data()

    np.testing.assert_allclose(data.probe_gain(0), gains)
    np.testing.assert_allclose(data.gain(0, "minimum"), gains.min(axis=2))
    # the ratio of averaged magnitudes is not the average of the gains
    assert not np.allclose(data.averaged_gain(0), gains.mean(axis=2))


def test_fit_minimum_across_probes():
    """The minimum objective favours the worst probe, unlike the average."""
    gains, data = two_probes_data()
    minimum = _fit(data)
    average = _fit(data, TwpaFrequencyOffsetFitParameters(objective="average"))

    def point(fit):
        i = list(OFFSETS).index(fit.offset[0])
        j = list(FREQUENCIES).index(fit.frequency[0])
        return gains[i, j]

    assert minimum.objective == "minimum" and average.objective == "average"
    assert minimum.frequency[0] != average.frequency[0]
    assert point(minimum).min() > point(average).min()
    np.testing.assert_allclose(minimum.probe_gain[0], point(minimum))
    assert minimum.gain[0] == pytest.approx(point(minimum).min())

    figures, report = _plot(data, minimum, 0)
    assert len(figures) == 3
    assert "Probe 7.0000 GHz Gain [dB]" in report
    assert "Probe 7.2000 GHz Gain [dB]" in report
    figures, report = _plot(data, average, 0)
    assert figures[0].data[0].colorbar.title.text == "Average TWPA Gain [dB]"
    assert "Average TWPA Gain [dB]" in report


def test_single_probe_objectives_coincide():
    data = landscape_data(OFFSETS, FREQUENCIES)

    np.testing.assert_allclose(data.gain(0, "minimum"), data.averaged_gain(0))
    fit = _fit(data)
    assert (
        fit.offset[0]
        == _fit(data, TwpaFrequencyOffsetFitParameters(objective="average")).offset[0]
    )
    figures, report = _plot(data, fit, 0)
    assert len(figures) == 1
    # the gain of the only probe is the aggregated one, but its reference is reported
    assert "Readout probe Gain" not in report
    assert "Readout probe reference drift [dB]" in report


def test_objective_fit_parameter(tmp_path):
    assert twpa_frequency_offset.load_fit_parameters(
        {"objective": "average"}
    ) == TwpaFrequencyOffsetFitParameters(objective="average")
    with pytest.raises(ValueError, match="Unknown gain objective"):
        TwpaFrequencyOffsetFitParameters(objective="maximum")

    _, data = two_probes_data()
    fit = _fit(data)
    fit.save(tmp_path)
    loaded = TwpaFrequencyOffsetResults.load(tmp_path)
    assert loaded.objective == "minimum"
    assert loaded.probe_gain == fit.probe_gain


def test_results_of_previous_versions():
    """Results without per-probe gains were optimizing the average gain."""
    results = TwpaFrequencyOffsetResults(
        frequency={0: 6.1e9}, offset={0: 0.5}, gain={0: 20.0}, power={0: -12.0}
    )
    assert 0 in results
    assert results.objective == "average"
    _, data = two_probes_data()
    figures, report = _plot(data, results, 0)
    assert figures[0].data[0].colorbar.title.text == "Average TWPA Gain [dB]"
    assert "Probe" not in report


def shared_pump_data(pumps=("pump", "pump", "other")):
    """One readout probe per target, as with omitted probes.

    The targets on ``pump`` have the gain peak at different pump frequencies, and the
    second one has a larger signal.
    """
    centers = [6.0e9, 6.02e9, 6.0e9]
    shifts = [0.0, 10.0, 0.0]
    references = [1.0, 10.0, 1.0]
    targets = list(range(len(pumps)))
    gains = {
        q: landscape(OFFSETS, FREQUENCIES, center=centers[q])[:, :, None] + shifts[q]
        for q in targets
    }
    magnitudes = {q: references[q] * 10 ** (gains[q] / 20) for q in targets}
    return gains, TwpaFrequencyOffsetData(
        data={
            q: np.stack([magnitudes[q], np.zeros_like(magnitudes[q])], axis=-1)
            for q in targets
        },
        reference_value={q: [[references[q], 0.0]] for q in targets},
        offset={q: list(OFFSETS) for q in targets},
        frequency={q: list(FREQUENCIES) for q in targets},
        power={q: 0.0 for q in targets},
        pump=dict(zip(targets, pumps)),
    )


def test_shared_pump_gain():
    """The gain is aggregated over the probes of all the targets on a pump."""
    gains, data = shared_pump_data()
    shared = np.concatenate([gains[0], gains[1]], axis=2)

    assert data.group(0) == data.group(1) == [0, 1]
    assert data.group(2) == [2]
    assert data.group_probes(1) == [(0, 0), (1, 0)]
    for qubit in (0, 1):
        np.testing.assert_allclose(data.probe_gain(qubit), shared)
        np.testing.assert_allclose(data.gain(qubit, "minimum"), shared.min(axis=2))
    np.testing.assert_allclose(data.probe_gain(2), gains[2])


def test_shared_pump_fit_and_plot():
    """A single point is selected for each pump, and plotted once."""
    gains, data = shared_pump_data()
    fit = _fit(data)

    assert fit.frequency[0] == fit.frequency[1]
    assert fit.offset[0] == fit.offset[1]
    assert fit.probe_gain[0] == fit.probe_gain[1]
    assert len(fit.probe_gain[0]) == 2
    assert fit.gain[0] == pytest.approx(min(fit.probe_gain[0]))
    # the worst resonator is better off than at the optimum of the other one alone
    i = list(OFFSETS).index(fit.offset[0])
    j = list(FREQUENCIES).index(fit.frequency[0])
    alone = _fit(shared_pump_data(pumps=("a", "b", "c"))[1])
    k = list(FREQUENCIES).index(alone.frequency[1])
    assert min(gains[0][i, j, 0], gains[1][i, j, 0]) > min(
        gains[0][i, k, 0], gains[1][i, k, 0]
    )
    assert len(fit.probe_gain[2]) == 1

    figures, report = _plot(data, fit, 0)
    # minimum and average maps, and per-probe maps, across the targets on the pump
    assert len(figures) == 3
    assert figures[0].data[0].colorbar.title.text == (
        "Minimum TWPA Gain across probes [dB]"
    )
    assert figures[1].data[0].colorbar.title.text == "Average TWPA Gain [dB]"
    assert [t.text for t in figures[2].layout.annotations] == [
        "Qubit 0 readout",
        "Qubit 1 readout",
    ]
    assert "shared by qubits 0, 1" in report
    assert "Qubit 0 readout Gain [dB]" in report
    assert "Qubit 1 readout Gain [dB]" in report

    figures, report = _plot(data, fit, 1)
    assert figures == []
    assert "Gain maps are reported for qubit 0" in report

    figures, report = _plot(data, fit, 2)
    assert len(figures) == 1
    assert "shared" not in report


def test_shared_pump_acquisition(platform):
    """The pump of each target is recorded, and it groups the targets."""
    targets = [0, 1]
    params = TwpaFrequencyOffsetParameters.load(
        {
            "amplitude": [0.1, 0.3, 0.1],
            "frequency": ["center", 10_000_000, 5_000_000],
            "nshots": 50,
        }
    )
    data = _acquisition(params, platform, targets)
    pumps = {
        q: platform.channels[platform.qubits[q].acquisition].twpa_pump for q in targets
    }
    assert data.pump == pumps
    for q in targets:
        assert data.group(q) == [t for t in targets if pumps[t] == pumps[q]]
        assert data.probe_gain(q).shape[2] == len(data.group(q))


def bracketed_data(before, after, zero=None):
    """A flat 10 dB landscape, whose reference changes from `before` to `after`.

    With `zero`, the null offset is scanned, giving the in-sweep reference `zero`.
    """
    offsets = [0.0, 0.5, 0.6] if zero is not None else [0.5, 0.6]
    magnitude = np.full((len(offsets), 3, 1), 10 ** (10 / 20))
    if zero is not None:
        magnitude[0] = zero
    return TwpaFrequencyOffsetData(
        data={0: np.stack([magnitude, np.zeros_like(magnitude)], axis=-1)},
        reference_value={0: [[[before, 0.0], [after, 0.0]]]},
        offset={0: offsets},
        frequency={0: [6.0e9, 6.01e9, 6.02e9]},
        power={0: 0.0},
    )


def test_bracketed_reference():
    """The reference is the average of the brackets, whose change is the drift."""
    data = bracketed_data(1.0, 10 ** (6 / 20))
    np.testing.assert_allclose(data.reference_drift(0), [6.0])
    reference = (1 + 10 ** (6 / 20)) / 2
    np.testing.assert_allclose(
        data.probe_gain(0)[..., 0], 10 - 20 * np.log10(reference)
    )
    fit = _fit(data)
    assert fit.reference_drift[0] == pytest.approx([6.0])
    _, report = _plot(data, fit, 0)
    assert "Unstable reference" in report and "+6.0 dB" in report


def test_previous_references_are_a_single_bracket():
    data = landscape_data(OFFSETS, FREQUENCIES)
    assert data.references(0).shape == (1, 1, 2)
    np.testing.assert_allclose(data.reference_drift(0), [0.0])


def test_suspicious_gain():
    data = bracketed_data(1e-2, 1e-2)
    fit = _fit(data)
    assert fit.gain[0] == pytest.approx(50.0)
    _, report = _plot(data, fit, 0)
    assert "Suspicious gain" in report and "Unstable reference" not in report


def test_zero_offset_reference():
    """The null offset row checks the pump-off reference, or replaces it."""
    data = bracketed_data(1.0, 1.0, zero=2.0)
    assert data.zero_offset(0) == 0
    np.testing.assert_allclose(data.zero_offset_error(0), [20 * np.log10(2)])

    pump_off = _fit(data)
    assert pump_off.zero_offset_error[0] == pytest.approx([20 * np.log10(2)])
    zero = _fit(data, TwpaFrequencyOffsetFitParameters(reference="zero_offset"))
    assert zero.reference == "zero_offset"
    assert zero.gain[0] == pytest.approx(pump_off.gain[0] - 20 * np.log10(2))
    # the reference only shifts the map: the point is the same
    assert (zero.frequency[0], zero.offset[0]) == (
        pump_off.frequency[0],
        pump_off.offset[0],
    )
    _, report = _plot(data, zero, 0)
    assert "zero offset error [dB]" in report


def test_zero_offset_reference_not_scanned():
    data = bracketed_data(1.0, 1.0)
    assert data.zero_offset(0) is None and data.zero_offset_error(0) is None
    assert _fit(data).zero_offset_error == {}
    with pytest.raises(ValueError, match="null pump offset"):
        _fit(data, TwpaFrequencyOffsetFitParameters(reference="zero_offset"))
    with pytest.raises(ValueError, match="Unknown reference"):
        TwpaFrequencyOffsetFitParameters(reference="other")


def test_acquisition_detunings(platform, mocker):
    params = TwpaFrequencyOffsetParameters.load(
        {
            "amplitude": [0.1, 0.3, 0.1],
            "frequency": ["center", 10_000_000, 5_000_000],
            "detunings": [-1e6, 2e6],
            "nshots": 10,
            "reference_shots": 40,
        }
    )
    spy = mocker.spy(platform, "execute")
    data = _acquisition(params, platform, [0, 1])
    assert data.detunings == [-1e6, 2e6]
    # each probe: reference, sweep, reference
    assert [len(c.args) for c in spy.call_args_list] == [1, 2, 1] * 2
    assert [c.kwargs["nshots"] for c in spy.call_args_list] == [40, 10, 40] * 2
    for index, detuning in enumerate([-1e6, 2e6]):
        probes = spy.call_args_list[3 * index + 1].kwargs["updates"][0]
        for q in (0, 1):
            assert probes[platform.qubits[q].probe]["frequency"] == (
                readout_frequency(q, platform) + detuning
            )
    assert probe_labels(data, 0) == [
        f"Qubit {q} Detuning {d} MHz" for q in (0, 1) for d in ("-1.000", "+2.000")
    ]
    _fit(data)


def test_pump_point():
    channel = _pump_platform(
        {"pump": IqChannel(path="8/o2", lo="pump/lo")},
        {
            "pump": IqConfig(frequency=6e9),
            "pump/lo": OscillatorConfig(frequency=5.9e9, power=-20.0),
        },
    )
    pump = Pumps(
        channel={0: "pump", 1: "pump"},
        config={0: IqConfig(frequency=6e9), 1: IqConfig(frequency=6e9)},
        power={0: -20.0, 1: -20.0},
    )
    frequency, updates = pump_point(channel, pump, None, 0.1)
    assert frequency == {0: 6e9, 1: 6e9}
    # the frequency on the pump, the power on its LO, both once for a shared pump
    assert updates == {
        "pump": {"frequency": 6e9},
        "pump/lo": {"power": pytest.approx(-40.0)},
    }
    frequency, updates = pump_point(channel, pump, 6.1e9, -1.0)
    assert frequency[0] == 6.1e9 and updates["pump/lo"]["power"] == -20.0
