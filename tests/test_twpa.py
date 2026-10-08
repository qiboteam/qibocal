"""Tests for TWPA calibration and frequency and offset sweeper protocols."""

import numpy as np
import pytest
from qibolab import (
    Acquisition,
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
    _fit as calibration_fit,
    _plot as calibration_plot,
    _twpa_scan,
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
    assert any(
        getattr(trace, "name", None) == "Optimal Point" and trace.showlegend is True
        for trace in fig.data
    )
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
        expected_power = np.arange(*params.power_range(configs[pump].power))
        expected_frequency = np.arange(*params.frequency_range(configs[pump].frequency))
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
            *(
                np.arange(*params.power_range(configs[pump].power))
                for pump in unique_pumps
            )
        )
        for frequencies in zip(
            *(
                np.arange(*params.frequency_range(configs[pump].frequency))
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
