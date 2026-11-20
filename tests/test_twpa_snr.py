"""Tests for the TWPA SNR improvement protocol."""

import numpy as np
import pytest
from qibolab import AveragingMode, OscillatorConfig

from qibocal.protocols import PROTOCOLS
from qibocal.protocols.twpa.snr import (
    TwpaSnrData,
    TwpaSnrFitParameters,
    TwpaSnrParameters,
    TwpaSnrResults,
    _acquisition,
    _fit,
    _plot,
    twpa_snr,
)
from qibocal.protocols.twpa.utils import PUMP_OFF_POWER, statistics
from qibocal.protocols.twpa.utils import update_point as _update

OFFSETS = np.round(np.arange(0.1, 1.0, 0.05), 2)
FREQUENCIES = 6.0e9 + np.arange(-30e6, 31e6, 5e6)


def iq(magnitude, spread):
    """Statistics of shots, as stored: the mean I and Q, and their spread."""
    magnitude = np.asarray(magnitude, dtype=float)
    return np.stack(
        [magnitude, np.zeros_like(magnitude), np.broadcast_to(spread, magnitude.shape)],
        axis=-1,
    )


def landscape(offsets, frequencies):
    """Gain growing with the pump power, until an over-pumping cliff at 0.8."""
    offsets, frequencies = np.asarray(offsets), np.asarray(frequencies)
    power = 20 * np.log10(np.abs(offsets))[:, np.newaxis]
    detuning = ((frequencies - 6.0e9) / 20e6)[np.newaxis, :] ** 2
    gain = 20 + 1.5 * power - 2 * detuning
    return np.where(np.abs(offsets)[:, np.newaxis] > 0.8, -5.0, gain)


def noisy_landscape_data():
    """Gain and added noise, whose SNR improvement peaks away from the gain's.

    The TWPA adds the same noise as the gain, plus a much larger one above an offset of
    0.5: the largest gain is not the largest SNR improvement.
    """
    gain = landscape(OFFSETS, FREQUENCIES)
    noise = 0.5 * gain + np.where(OFFSETS > 0.5, 15.0, 0.0)[:, np.newaxis]
    return (
        gain,
        noise,
        TwpaSnrData(
            data={0: iq(10 ** (gain / 20), 10 ** (noise / 20))[:, :, None]},
            reference_value={0: [[1.0, 0.0, 1.0]]},
            offset={0: list(OFFSETS)},
            frequency={0: list(FREQUENCIES)},
            power={0: 0.0},
        ),
    )


def test_registration():
    assert PROTOCOLS["twpa_snr"] is twpa_snr


def test_parameters_validation():
    params = TwpaSnrParameters.load(
        {
            "amplitude": [0.0, 0.5, 0.1],
            "frequency": ["center", 20_000_000, 2_000_000],
            "reference_shots": 1000,
            "nshots": 100,
        }
    )
    assert params.reference_shots == 1000
    assert params.max_values > 0
    with pytest.raises(ValueError, match="maximum number of shots"):
        TwpaSnrParameters(amplitude=[0.1, 0.2, 0.1], frequency=[1, 2, 1], max_values=0)
    with pytest.raises(ValueError, match="Unknown objective"):
        TwpaSnrFitParameters(objective="fidelity")


def test_statistics():
    rng = np.random.default_rng(0)
    shots = rng.normal([3.0, -1.0], [2.0, 4.0], size=(20_000, 5, 2))
    stats = statistics(shots)
    assert stats.shape == (5, 3)
    np.testing.assert_allclose(stats[:, :2], [[3.0, -1.0]] * 5, atol=0.15)
    # the root mean square of the spreads of I and Q
    np.testing.assert_allclose(stats[:, 2], np.sqrt((2.0**2 + 4.0**2) / 2), rtol=0.02)
    with pytest.raises(ValueError, match="two shots"):
        statistics(shots[:1])


def test_improvement_is_gain_net_of_noise():
    """Signal doubles, spread quadruples: 6 dB of gain, 12 dB of noise."""
    data = TwpaSnrData(
        data={0: iq([[[2.0]]], 4.0)},
        reference_value={0: [[1.0, 0.0, 1.0]]},
        offset={0: [0.5]},
        frequency={0: [6e9]},
    )
    np.testing.assert_allclose(data.probe_gain(0), [[[20 * np.log10(2)]]])
    np.testing.assert_allclose(data.probe_noise(0), [[[20 * np.log10(4)]]])
    np.testing.assert_allclose(data.probe_snr(0), [[[20 * np.log10(0.5)]]])
    # the reference is not unit: only ratios matter
    data.reference_value = {0: [[3.0, 0.0, 0.5]]}
    np.testing.assert_allclose(data.probe_gain(0), [[[20 * np.log10(2 / 3)]]])
    np.testing.assert_allclose(
        data.probe_snr(0), [[[20 * np.log10(2 / 4 / (3 / 0.5))]]]
    )


@pytest.mark.parametrize(
    "reference",
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [np.nan, 0.0, 1.0], [1.0, 0.0, np.inf]],
)
def test_invalid_reference(reference):
    data = TwpaSnrData(
        data={0: iq([[[2.0]]], 1.0)},
        reference_value={0: [reference]},
        offset={0: [0.5]},
        frequency={0: [6e9]},
    )
    with np.errstate(divide="raise", invalid="raise"):
        for operation in (lambda: _fit(data), lambda: _plot(data, None, 0)):
            with pytest.raises(ValueError, match="TWPA-off reference"):
                operation()


def test_objectives_select_different_points():
    gain, noise, data = noisy_landscape_data()
    snr = _fit(data)
    by_gain = _fit(data, TwpaSnrFitParameters(objective="gain"))

    assert snr.objective == "snr" and by_gain.objective == "gain"
    # the gain peaks below the cliff, the SNR improvement below the noisy region
    assert by_gain.offset[0] > 0.5 >= snr.offset[0]

    def point(fit):
        return (
            list(OFFSETS).index(fit.offset[0]),
            list(FREQUENCIES).index(fit.frequency[0]),
        )

    i, j = point(snr)
    assert snr.improvement[0] == pytest.approx(gain[i, j] - noise[i, j])
    assert snr.probe_gain[0] == pytest.approx([gain[i, j]])
    assert snr.probe_noise[0] == pytest.approx([noise[i, j]])
    assert snr.probe_snr[0] == pytest.approx([gain[i, j] - noise[i, j]])
    assert snr.power[0] == pytest.approx(20 * np.log10(snr.offset[0]))
    # the SNR improvement of the point selected by the gain is lower
    i, j = point(by_gain)
    assert gain[i, j] - noise[i, j] < snr.improvement[0]


def test_plot_and_report():
    _, _, data = noisy_landscape_data()
    fit = _fit(data)
    figures, report = _plot(data, fit, 0)
    # the optimized map, then the other one; a single probe has no per-probe maps
    assert [f.data[0].colorbar.title.text for f in figures] == [
        "Minimum SNR improvement across probes [dB]",
        "Minimum TWPA Gain across probes [dB]",
    ]
    assert any(trace.name == "Optimal Point" for trace in figures[0].data)
    assert figures[0].layout.yaxis2.title.text == "Pump power [dB]"
    for label in ("Pump Frequency [Hz]", "SNR improvement", "Gain", "Added noise"):
        assert label in report

    figures, report = _plot(data, None, 0)
    assert len(figures) == 2 and "<table" not in report

    by_gain = _fit(data, TwpaSnrFitParameters(objective="gain"))
    figures, _ = _plot(data, by_gain, 0)
    assert figures[0].data[0].colorbar.title.text.startswith("Minimum TWPA Gain")


def test_candidate_at_scan_edge():
    """A maximum at the edge of the scan is a candidate, reported but not applied."""
    offsets, frequencies = [0.1, 0.2, 0.3], [5.99e9, 6.0e9, 6.01e9]
    improvement = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 9.0]])
    data = TwpaSnrData(
        data={0: iq(10 ** (improvement / 20), 1.0)[:, :, None]},
        reference_value={0: [[1.0, 0.0, 1.0]]},
        offset={0: offsets},
        frequency={0: frequencies},
        power={0: 0.0},
    )
    fit = _fit(data, TwpaSnrFitParameters(neighbourhood=0))
    assert fit.is_candidate(0)
    _, report = _plot(data, fit, 0)
    assert "Fit failed" in report and "pump amplitude and pump frequency" in report


def test_update(platform):
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    results = TwpaSnrResults(
        frequency={0: 6.1e9},
        offset={0: 0.5},
        improvement={0: 5.0},
        power={0: -12.0},
        candidate={0: False},
    )
    _update(results, platform, 0)
    assert platform.config(pump) == OscillatorConfig(frequency=6.1e9, power=-12.0)

    results.candidate = {0: True}
    results.frequency = {0: 7.0e9}
    _update(results, platform, 0)
    assert platform.config(pump).frequency == 6.1e9


@pytest.mark.parametrize("probes", [None, [7e9, 7.2e9]])
@pytest.mark.parametrize("max_values", [1, 150, 10**9])
def test_acquisition_batches(platform, mocker, probes, max_values):
    """The offsets are split in batches, bounding the shots held by an execution."""
    params = TwpaSnrParameters.load(
        {
            "amplitude": [0.1, 0.6, 0.1],
            "frequency": ["center", 10_000_000, 2_000_000],
            "probes": probes,
            "nshots": 10,
            "reference_shots": 25,
            "max_values": max_values,
        }
    )
    spy = mocker.spy(platform, "execute")
    data = _acquisition(params, platform, [0, 1])

    nprobes = 1 if probes is None else len(probes)
    noffsets, nfrequencies = 5, 5
    rows = max(1, max_values // (params.nshots * nfrequencies))
    nbatches = -(-noffsets // rows)
    # for each probe, the sweep of each batch, bracketed by two references
    assert spy.call_count == nprobes * (nbatches + 2)
    for call in spy.call_args_list:
        assert call.kwargs["averaging_mode"] is AveragingMode.SINGLESHOT
    swept = [len(call.args) > 1 for call in spy.call_args_list]
    assert swept == nprobes * ([False] + [True] * nbatches + [False])
    references = [c for c, s in zip(spy.call_args_list, swept) if not s]
    assert all(call.kwargs["nshots"] == 25 for call in references)
    sweeps = [c for c, s in zip(spy.call_args_list, swept) if s]
    assert all(call.kwargs["nshots"] == 10 for call in sweeps)
    sizes = [len(call.args[1][0][0].values) for call in sweeps[:nbatches]]
    assert sum(sizes) == noffsets and max(sizes) <= rows

    assert isinstance(data, TwpaSnrData)
    for qubit in (0, 1):
        assert data[qubit].shape == (noffsets, nfrequencies, nprobes, 3)
        assert data.references(qubit).shape == (nprobes, 2, 3)
        assert data.offset[qubit] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
        assert len(data.frequency[qubit]) == nfrequencies
    assert data.group(0) == [0, 1]


def test_batched_acquisition_is_the_unbatched_one(platform, mocker):
    """Batching only splits the work: the pump offsets are covered in order."""
    params = {
        "amplitude": [0.1, 0.6, 0.1],
        "frequency": ["center", 10_000_000, 2_000_000],
        "nshots": 10,
    }
    spy = mocker.spy(platform, "execute")
    _acquisition(TwpaSnrParameters.load({**params, "max_values": 50}), platform, [0])
    offsets = np.concatenate(
        [c.args[1][0][0].values for c in spy.call_args_list if len(c.args) > 1]
    )
    np.testing.assert_allclose(offsets, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_acquisition_reference_pump_off(platform, mocker):
    params = TwpaSnrParameters.load(
        {
            "amplitude": [0.1, 0.3, 0.1],
            "frequency": ["center", 10_000_000, 5_000_000],
            "nshots": 10,
        }
    )
    spy = mocker.spy(platform, "execute")
    _acquisition(params, platform, [0])
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    assert spy.call_args_list[0].kwargs["updates"][1] == {
        pump: {"power": PUMP_OFF_POWER}
    }
    # by default, as many shots as the scan
    assert spy.call_args_list[0].kwargs["nshots"] == 10


def test_acquisition_default_shots(platform, mocker):
    """Without shots, the batches are sized on the platform ones."""
    params = TwpaSnrParameters.load(
        {"amplitude": [0.1, 0.3, 0.1], "frequency": ["center", 4_000_000, 2_000_000]}
    )
    assert params.nshots is None
    spy = mocker.spy(platform, "execute")
    _acquisition(params, platform, [0])
    assert {call.kwargs["nshots"] for call in spy.call_args_list} == {
        platform.settings.nshots
    }


def test_acquisition_invalid_amplitude(platform):
    params = TwpaSnrParameters.load(
        {"amplitude": [0.5, 2.0, 0.5], "frequency": ["center", 4_000_000, 2_000_000]}
    )
    with pytest.raises(ValueError, match="between -1 and 1"):
        _acquisition(params, platform, [0])


def test_acquisition_fit_plot_and_serialization(platform, tmp_path):
    params = TwpaSnrParameters.load(
        {
            "amplitude": [0.1, 0.5, 0.1],
            "frequency": ["center", 10_000_000, 2_000_000],
            "probes": [7e9, 7.2e9],
            "nshots": 20,
        }
    )
    data = _acquisition(params, platform, [0, 1])
    fit = _fit(data)
    assert fit.frequency[0] == fit.frequency[1] and fit.offset[0] == fit.offset[1]
    assert len(fit.probe_snr[0]) == 4  # two probes of each of the two targets
    figures, report = _plot(data, fit, 0)
    # objective, other, and per-probe maps
    assert len(figures) == 3
    assert _plot(data, fit, 1)[0] == []
    assert "shared by qubits 0, 1" in report

    data.save(tmp_path)
    loaded = TwpaSnrData.load(tmp_path)
    for qubit in (0, 1):
        np.testing.assert_array_equal(data[qubit], loaded[qubit])
        assert data.reference_value[qubit] == loaded.reference_value[qubit]
        assert data.offset[qubit] == loaded.offset[qubit]
    assert data.probes == loaded.probes
    fit.save(tmp_path)
    loaded_fit = TwpaSnrResults.load(tmp_path)
    assert loaded_fit.frequency == fit.frequency
    assert loaded_fit.probe_snr == fit.probe_snr


def test_bracketed_reference():
    """Magnitudes are averaged over the brackets, spreads in quadrature."""
    data = TwpaSnrData(
        data={0: iq([[[2.0]]], 4.0)},
        reference_value={0: [[[1.0, 0.0, 1.0], [0.0, 3.0, 7**0.5]]]},
        offset={0: [0.5]},
        frequency={0: [6e9]},
    )
    assert data.references(0).shape == (1, 2, 3)
    np.testing.assert_allclose(data.reference_drift(0), [20 * np.log10(3)])
    np.testing.assert_allclose(data.probe_gain(0), [[[0.0]]])
    np.testing.assert_allclose(data.probe_noise(0), [[[20 * np.log10(2)]]])
    fit = _fit(data)
    assert fit.reference_drift[0] == pytest.approx([20 * np.log10(3)])
    _, report = _plot(data, fit, 0)
    assert "Unstable reference" in report
