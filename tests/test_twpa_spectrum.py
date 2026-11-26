"""Tests for the TWPA gain spectrum protocol."""

import numpy as np
import pytest
from qibolab import AveragingMode, Parameter

from qibocal.protocols import PROTOCOLS
from qibocal.protocols.twpa.spectrum import (
    TwpaSpectrumData,
    TwpaSpectrumFitParameters,
    TwpaSpectrumParameters,
    TwpaSpectrumResults,
    _acquisition,
    _fit,
    _plot,
    dip,
    twpa_spectrum,
)
from qibocal.protocols.twpa.utils import PUMP_OFF_POWER
from qibocal.protocols.utils import readout_frequency

READOUT = 7.0e9
FREQUENCIES = READOUT + np.arange(-10e6, 10.01e6, 0.1e6)
LINEWIDTH = 0.5e6


def notch(frequencies, center, depth=0.95, linewidth=LINEWIDTH):
    """Transmission through a line with a notch resonance."""
    return np.abs(1 - depth / (1 + 2j * (frequencies - center) / linewidth))


def spectrum_data(gain=15.0, shift=0.0, jitter=(0.0,), ripple=0.0):
    """Spectra of a resonance, which the pump may shift.

    The TWPA-off resonance lies at the readout frequency, moved by each `jitter`, one per
    repetition. The gain may ripple in the probe frequency, by `ripple` [dB].
    """
    centers = READOUT + np.array(jitter)
    amplitude = 10 ** (
        (gain + ripple * np.sin(2 * np.pi * (FREQUENCIES - READOUT) / 40e6)) / 20
    )
    off = np.array([notch(FREQUENCIES, c) for c in centers])
    on = np.array([amplitude * notch(FREQUENCIES, c + shift) for c in centers])

    def iq(values):
        return np.stack([values, np.zeros_like(values)], axis=-1)

    return TwpaSpectrumData(
        data={0: iq(on)},
        reference_value={0: iq(off).tolist()},
        probe_frequency={0: FREQUENCIES.tolist()},
        readout={0: READOUT},
        frequency={0: 6e9},
        time=list(range(len(centers))),
    )


def test_registration():
    assert PROTOCOLS["twpa_spectrum"] is twpa_spectrum


def test_parameters_validation():
    params = TwpaSpectrumParameters.load(
        {"probe_frequency": ["center", 20e6, 0.1e6], "repeats": 2}
    )
    assert params.repeats == 2 and params.offset == 1.0
    with pytest.raises(ValueError, match="one repetition"):
        TwpaSpectrumParameters(probe_frequency=["center", 1e6, 1e5], repeats=0)
    with pytest.raises(ValueError, match="exclusion"):
        TwpaSpectrumFitParameters(exclusion=-1.0)


def test_dip():
    centers = READOUT + np.array([-1.23e6, 0.0, 0.47e6])
    spectra = np.array([notch(FREQUENCIES, c) for c in centers])
    # refined below the grid step
    np.testing.assert_allclose(dip(FREQUENCIES, spectra), centers, atol=0.02e6)
    # at the edge, not refined
    edge = np.array([np.linspace(0, 1, 5)])
    assert dip(np.arange(5.0), edge) == [0.0]


def test_steady_resonance():
    """Without a shift, the gain is the amplifier one, at the readout as well."""
    fit = _fit(spectrum_data(gain=15.0))
    assert fit.baseline_gain[0] == pytest.approx(15.0)
    assert fit.readout_gain[0] == pytest.approx(15.0)
    assert fit.distortion[0] == pytest.approx(0.0, abs=1e-9)
    assert fit.dip_shift[0] == pytest.approx(0.0, abs=1e3)
    assert fit.dip[0] == pytest.approx(READOUT, abs=1e3)
    # a 95% deep notch is 26 dB deep
    assert fit.readout_depth[0] == pytest.approx(-20 * np.log10(0.05), abs=0.1)
    assert fit.readout_gain_std[0] == 0.0


def test_pump_shifting_the_resonance():
    """A pump moving the resonance inflates the gain at the readout frequency."""
    fit = _fit(spectrum_data(gain=15.0, shift=1e6))
    assert fit.dip_shift[0] == pytest.approx(1e6, abs=0.02e6)
    assert fit.baseline_gain[0] == pytest.approx(15.0, abs=0.1)
    # the probe is now off the resonance: the dip depth is added to the gain
    assert fit.readout_gain[0] > 35
    assert fit.distortion[0] == pytest.approx(fit.readout_gain[0] - 15.0, abs=0.1)


def test_jittering_resonance():
    """A resonance moving across repetitions spoils the gain at the readout only."""
    jitter = [0.0, 0.3e6, -0.2e6, 0.1e6]
    fit = _fit(spectrum_data(gain=15.0, shift=0.5e6, jitter=jitter))
    assert fit.dip_jitter[0] == pytest.approx(np.std(jitter, ddof=1), rel=0.1)
    assert fit.readout_gain_std[0] > 3
    assert fit.baseline_gain_std[0] < 0.1


def test_baseline_is_robust_to_ripple():
    fit = _fit(spectrum_data(gain=15.0, ripple=1.0))
    assert fit.baseline_gain[0] == pytest.approx(15.0, abs=0.5)


def test_exclusion_covering_the_range():
    fit = _fit(spectrum_data(), TwpaSpectrumFitParameters(exclusion=1e9))
    assert 0 not in fit.baseline_gain
    figures, report = _plot(spectrum_data(), fit, 0)
    assert len(figures) == 2 and report == ""


def test_plot_and_report():
    data = spectrum_data(gain=15.0, shift=1e6, jitter=[0.0, 0.1e6])
    fit = _fit(data)
    figures, report = _plot(data, fit, 0)
    assert len(figures) == 2
    # TWPA off and on, for each repetition
    assert len(figures[0].data) == 2 * 2
    for label in ("Baseline gain [dB]", "Resonance shift by the pump [MHz]", "depth"):
        assert label in report
    figures, report = _plot(data, None, 0)
    assert len(figures) == 2 and report == ""


def test_acquisition(platform, mocker, tmp_path):
    params = TwpaSpectrumParameters.load(
        {
            "probe_frequency": ["center", 4e6, 1e6],
            "repeats": 2,
            "offset": 0.5,
            "nshots": 10,
            "reference_shots": 30,
        }
    )
    spy = mocker.spy(platform, "execute")
    targets = [0, 1]
    data = _acquisition(params, platform, targets)

    assert spy.call_count == 2 * 2
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    power = platform.config(pump).power
    for index, call in enumerate(spy.call_args_list):
        assert call.kwargs["averaging_mode"] is AveragingMode.CYCLIC
        (sweepers,) = call.args[1]
        assert [s.parameter for s in sweepers] == [Parameter.frequency] * 2
        assert [s.channels for s in sweepers] == [
            [platform.qubits[q].probe] for q in targets
        ]
        (update,) = call.kwargs["updates"]
        if index % 2 == 0:
            assert update == {pump: {"power": PUMP_OFF_POWER}}
            assert call.kwargs["nshots"] == 30
        else:
            assert update[pump]["power"] == pytest.approx(power + 20 * np.log10(0.5))
            assert call.kwargs["nshots"] == 10

    for q in targets:
        readout = readout_frequency(q, platform)
        assert data.readout[q] == readout
        np.testing.assert_allclose(
            data.probe_frequency[q], readout + np.arange(-2e6, 2e6, 1e6)
        )
        assert data[q].shape == (2, 4, 2)

    fit = _fit(data, TwpaSpectrumFitParameters(exclusion=0.5e6))
    _plot(data, fit, 0)
    data.save(tmp_path)
    loaded = TwpaSpectrumData.load(tmp_path)
    np.testing.assert_array_equal(loaded[0], data[0])
    assert loaded.readout == data.readout
    fit.save(tmp_path)
    assert TwpaSpectrumResults.load(tmp_path).dip_shift == fit.dip_shift
