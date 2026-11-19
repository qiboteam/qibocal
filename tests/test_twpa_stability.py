"""Tests for the TWPA stability validation protocol."""

import numpy as np
import pytest
from qibolab import AveragingMode

from qibocal.protocols import PROTOCOLS
from qibocal.protocols.twpa.stability import (
    TwpaStabilityData,
    TwpaStabilityParameters,
    TwpaStabilityResults,
    _acquisition,
    _fit,
    _plot,
    twpa_stability,
)
from qibocal.protocols.twpa.utils import PUMP_OFF_POWER


def stability_data(on, off):
    """Data from the magnitudes with the TWPA on and off, with shape ``(repeats,)``."""

    def signal(magnitudes):
        return np.stack([magnitudes, np.zeros_like(magnitudes)], axis=-1)[:, None, :]

    return TwpaStabilityData(
        data={0: signal(np.asarray(on, dtype=float))},
        reference_value={0: signal(np.asarray(off, dtype=float)).tolist()},
        frequency={0: 6e9},
        time=list(range(len(on))),
    )


def test_registration():
    assert PROTOCOLS["twpa_stability"] is twpa_stability


def test_parameters_validation():
    params = TwpaStabilityParameters.load({"repeats": 5, "offset": 0.8})
    assert params.repeats == 5 and params.offset == 0.8 and params.frequency is None
    with pytest.raises(ValueError, match="two repetitions"):
        TwpaStabilityParameters(repeats=1)


def test_gain_of_each_repetition():
    data = stability_data(on=[10.0, 20.0, 40.0], off=[1.0, 2.0, 2.0])
    np.testing.assert_allclose(data.gain(0)[:, 0], 20 * np.log10([10, 10, 20]))
    # with the mean reference, only the signal is left
    np.testing.assert_allclose(
        data.common_gain(0)[:, 0], 20 * np.log10(np.array([10, 20, 40]) / (5 / 3))
    )


def test_unstable_reference():
    """A drifting reference spoils the gain, unless it is a single common one."""
    drift = 10 ** (np.array([-6.0, 3.0, 0.0, 6.0, -3.0, 0.0]) / 20)
    fit = _fit(stability_data(on=np.full(6, 10.0), off=drift))

    expected = 20 * np.log10(drift)
    assert fit.reference_std[0][0] == pytest.approx(np.std(expected, ddof=1))
    assert fit.pumped_std[0][0] == pytest.approx(0.0, abs=1e-9)
    # the gain follows the reference, which a common one would remove
    assert fit.gain_std[0][0] == pytest.approx(np.std(expected, ddof=1))
    assert fit.common_gain_std[0][0] == pytest.approx(0.0, abs=1e-9)


def test_unstable_pump():
    """A drifting signal is not cured by the reference: the pump is to blame."""
    drift = 10 ** (np.array([-6.0, 3.0, 0.0, 6.0, -3.0, 0.0]) / 20)
    fit = _fit(stability_data(on=10.0 * drift, off=np.full(6, 2.0)))

    assert fit.reference_std[0][0] == pytest.approx(0.0, abs=1e-9)
    assert fit.pumped_std[0][0] > 3
    assert fit.gain_std[0][0] == pytest.approx(fit.common_gain_std[0][0])
    # no reference fluctuation: correlation is not defined, and reported as null
    assert fit.correlation[0][0] == 0.0


def test_correlated_fluctuations():
    """Signals moving together, e.g. a common-mode drift of the readout chain."""
    drift = 10 ** (np.array([-6.0, 3.0, 0.0, 6.0, -3.0, 0.0]) / 20)
    fit = _fit(stability_data(on=10.0 * drift, off=2.0 * drift))
    assert fit.correlation[0][0] == pytest.approx(1.0)
    # the gain is immune to it
    assert fit.gain_std[0][0] == pytest.approx(0.0, abs=1e-9)
    assert fit.gain[0][0] == pytest.approx(20 * np.log10(5))


@pytest.mark.parametrize("off", [0.0, np.nan, np.inf])
def test_invalid_reference(off):
    data = stability_data(on=[1.0, 2.0, 3.0], off=[1.0, off, 1.0])
    with np.errstate(divide="raise", invalid="raise"):
        for operation in (lambda: _fit(data), lambda: _plot(data, None, 0)):
            with pytest.raises(ValueError, match="TWPA-off reference magnitude"):
                operation()


@pytest.mark.parametrize("probes", [None, [7e9, 7.2e9]])
def test_acquisition(platform, mocker, probes):
    params = TwpaStabilityParameters.load(
        {"repeats": 3, "offset": 0.7, "nshots": 10, "probes": probes}
    )
    spy = mocker.spy(platform, "execute")
    data = _acquisition(params, platform, [0, 1])
    nprobes = 1 if probes is None else len(probes)

    # per repetition, a reference and a signal for each probe
    assert spy.call_count == 3 * 2 * nprobes
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    for index, call in enumerate(spy.call_args_list):
        assert call.kwargs["averaging_mode"] is AveragingMode.CYCLIC
        is_reference = index % (2 * nprobes) < nprobes
        _, other = call.kwargs["updates"]
        if is_reference:
            assert other == {pump: {"power": PUMP_OFF_POWER}}
            assert len(call.args) == 1
        else:
            # the pump is at the configured frequency, with the offset swept
            assert other == {pump: {"frequency": data.frequency[0]}}
            (sweeper,) = call.args[1][0]
            assert sweeper.values.tolist() == [0.7]

    assert data.frequency[0] == platform.config(pump).frequency
    assert data.offset == 0.7
    assert data.probes == probes
    for qubit in (0, 1):
        assert data[qubit].shape == (3, nprobes, 2)
        assert data.reference_array(qubit).shape == (3, nprobes, 2)
        assert data.nprobes(qubit) == nprobes
    assert len(data.time) == 3 and data.time == sorted(data.time)


def test_acquisition_frequency_and_invalid_offset(platform, mocker):
    params = TwpaStabilityParameters.load(
        {"repeats": 2, "frequency": 6.5e9, "nshots": 10}
    )
    spy = mocker.spy(platform, "execute")
    data = _acquisition(params, platform, [0])
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    assert data.frequency == {0: 6.5e9}
    assert spy.call_args_list[1].kwargs["updates"][1] == {pump: {"frequency": 6.5e9}}

    with pytest.raises(ValueError, match="between -1 and 1"):
        _acquisition(
            TwpaStabilityParameters.load({"repeats": 2, "offset": 1.5}), platform, [0]
        )


def test_fit_plot_and_serialization(platform, tmp_path):
    params = TwpaStabilityParameters.load(
        {"repeats": 4, "nshots": 10, "probes": [7e9, 7.2e9]}
    )
    data = _acquisition(params, platform, [0])
    fit = _fit(data)
    assert len(fit.gain[0]) == 2
    for stat in (
        fit.gain_std,
        fit.common_gain_std,
        fit.pumped_std,
        fit.reference_std,
        fit.correlation,
    ):
        assert len(stat[0]) == 2

    figures, report = _plot(data, fit, 0)
    assert len(figures) == 2
    # the gain, with its own and common reference, of each probe
    assert len(figures[0].data) == 4
    for label in ("Probe 7.0000 GHz Mean gain", "Probe 7.2000 GHz Gain std", "On/off"):
        assert label in report
    figures, report = _plot(data, None, 0)
    assert len(figures) == 2 and report == ""

    data.save(tmp_path)
    loaded = TwpaStabilityData.load(tmp_path)
    np.testing.assert_array_equal(data[0], loaded[0])
    np.testing.assert_allclose(data.reference_array(0), loaded.reference_array(0))
    assert loaded.time == data.time and loaded.offset == data.offset
    fit.save(tmp_path)
    assert TwpaStabilityResults.load(tmp_path).gain_std == fit.gain_std
