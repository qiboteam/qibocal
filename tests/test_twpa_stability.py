"""Tests for the TWPA stability validation protocol."""

import numpy as np
import pytest
from qibolab import AveragingMode

from qibocal.protocols import PROTOCOLS
from qibocal.protocols.twpa.stability import (
    DB,
    TwpaStabilityData,
    TwpaStabilityParameters,
    TwpaStabilityResults,
    _acquisition,
    _diagnosis,
    _fit,
    _plot,
    twpa_stability,
)
from qibocal.protocols.twpa.utils import PUMP_OFF_POWER

DRIFT = 10 ** (np.array([-6.0, 3.0, 0.0, 6.0, -3.0, 0.0]) / 20)
"""Relative drift of a signal, over six repetitions, by several dB."""


def statistics(magnitudes, spread=1e-6, phases=0.0):
    """Shots statistics from the magnitudes, with shape ``(repeats, 1, 3)``.

    The default spread is negligible: every fluctuation is a drift.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    phases = np.broadcast_to(np.radians(phases), magnitudes.shape)
    return np.stack(
        [
            magnitudes * np.cos(phases),
            magnitudes * np.sin(phases),
            np.broadcast_to(spread, magnitudes.shape),
        ],
        axis=-1,
    )[:, None, :]


def stability_data(on, off, spread=1e-6, off_phases=0.0, shots=100):
    """Data from the magnitudes with the TWPA on and off, with shape ``(repeats,)``."""
    return TwpaStabilityData(
        data={0: statistics(on, spread)},
        reference_value={0: statistics(off, spread, off_phases).tolist()},
        frequency={0: 6e9},
        time=list(range(len(on))),
        shots=shots,
        reference_shots=shots,
    )


def test_registration():
    assert PROTOCOLS["twpa_stability"] is twpa_stability


def test_parameters_validation():
    params = TwpaStabilityParameters.load({"repeats": 5, "offset": 0.8})
    assert params.repeats == 5 and params.offset == 0.8 and params.frequency is None
    with pytest.raises(ValueError, match="two repetitions"):
        TwpaStabilityParameters(repeats=1)
    with pytest.raises(ValueError, match="interval"):
        TwpaStabilityParameters(interval=-1.0)
    with pytest.raises(ValueError, match="either 'probes' or 'detunings'"):
        TwpaStabilityParameters(probes=[7e9], detunings=[0.0])


def test_gain_of_each_repetition():
    data = stability_data(on=[10.0, 20.0, 40.0], off=[1.0, 2.0, 2.0])
    np.testing.assert_allclose(data.gain(0)[:, 0], 20 * np.log10([10, 10, 20]))
    # with the mean reference, only the signal is left
    np.testing.assert_allclose(
        data.common_gain(0)[:, 0], 20 * np.log10(np.array([10, 20, 40]) / (5 / 3))
    )


def test_unstable_reference():
    """A drifting reference spoils the gain, unless it is a single common one."""
    fit = _fit(stability_data(on=np.full(6, 10.0), off=DRIFT))

    expected = 20 * np.log10(DRIFT)
    assert fit.reference_std[0][0] == pytest.approx(np.std(expected, ddof=1))
    assert fit.pumped_std[0][0] == pytest.approx(0.0, abs=1e-9)
    # the gain follows the reference, which a common one would remove
    assert fit.gain_std[0][0] == pytest.approx(np.std(expected, ddof=1))
    assert fit.common_gain_std[0][0] == pytest.approx(0.0, abs=1e-9)
    assert fit.reference_excess[0][0] > 1e3
    assert fit.diagnosis[0][0] == "reference drift"


def test_unstable_pump():
    """A drifting signal is not cured by the reference: the pump is to blame."""
    fit = _fit(stability_data(on=10.0 * DRIFT, off=np.full(6, 2.0)))

    assert fit.reference_std[0][0] == pytest.approx(0.0, abs=1e-9)
    assert fit.pumped_std[0][0] > 3
    assert fit.gain_std[0][0] == pytest.approx(fit.common_gain_std[0][0])
    # no reference fluctuation: correlation is not defined, and reported as null
    assert fit.correlation[0][0] == 0.0
    assert fit.diagnosis[0][0].startswith("pumped signal drift")


def test_correlated_fluctuations():
    """Signals moving together, e.g. a common-mode drift of the readout chain."""
    fit = _fit(stability_data(on=10.0 * DRIFT, off=2.0 * DRIFT))
    assert fit.correlation[0][0] == pytest.approx(1.0)
    # the gain is immune to it
    assert fit.gain_std[0][0] == pytest.approx(0.0, abs=1e-9)
    assert fit.gain[0][0] == pytest.approx(20 * np.log10(5))
    assert fit.diagnosis[0][0].startswith("common drift")


def test_shot_noise():
    """Fluctuations as large as the ones expected from the shots are not drifts."""
    rng = np.random.default_rng(42)
    repeats, nshots, spread = 400, 100, 0.5
    # the mean of the shots fluctuates by the spread, over the root of their number
    noise = spread / np.sqrt(nshots)
    on = 10.0 + noise * rng.standard_normal(repeats)
    off = 2.0 + noise * rng.standard_normal(repeats)
    data = stability_data(on=on, off=off, spread=spread, shots=nshots)

    noise_on, noise_off = data.shot_noise(0)
    assert noise_on[0] == pytest.approx(noise / 10, rel=1e-2)
    assert noise_off[0] == pytest.approx(noise / 2, rel=1e-2)
    fit = _fit(data)
    assert fit.pumped_excess[0][0] == pytest.approx(1.0, abs=0.15)
    assert fit.reference_excess[0][0] == pytest.approx(1.0, abs=0.15)
    assert fit.reference_std[0][0] == pytest.approx(DB * noise / 2, rel=0.15)
    assert fit.reference_snr[0][0] == pytest.approx(20 * np.log10(2 / spread), abs=0.1)
    assert fit.diagnosis[0][0] == "shot noise"


def test_reference_phase_drift():
    """A phase drifting with a steady magnitude, as on top of a moving resonance."""
    phases = np.array([-20.0, 10.0, 0.0, 20.0, -10.0, 0.0])
    data = stability_data(on=np.full(6, 10.0), off=np.full(6, 2.0), off_phases=phases)
    _, phase_off = data.phase_fluctuations(0)
    np.testing.assert_allclose(phase_off[:, 0], phases, atol=1e-9)
    fit = _fit(data)
    assert fit.reference_phase_std[0][0] == pytest.approx(np.std(phases, ddof=1))
    assert fit.reference_std[0][0] == pytest.approx(0.0, abs=1e-9)
    assert "reference phase drift" in fit.diagnosis[0][0]


def test_phase_fluctuations_across_the_branch_cut():
    """Phases are compared to their circular mean, unaffected by the wrapping."""
    phases = np.array([179.0, -179.0, 180.0, -180.0])
    data = stability_data(on=np.ones(4), off=np.ones(4), off_phases=phases)
    _, phase_off = data.phase_fluctuations(0)
    assert np.max(np.abs(phase_off)) < 1.5


@pytest.mark.parametrize(
    "args, verdict",
    [
        ((1.0, 1.0, 0.1, 0.1, 0.0, 1.0), "shot noise"),
        ((1.0, 10.0, 0.1, 3.0, 0.0, 1.0), "reference drift"),
        ((10.0, 1.0, 3.0, 0.1, 0.0, 1.0), "pumped signal drift (pump, or TWPA)"),
        (
            (10.0, 10.0, 3.0, 3.0, 0.9, 1.0),
            "common drift (readout chain, or resonator)",
        ),
        ((10.0, 10.0, 1.0, 3.0, 0.0, 1.0), "reference drift"),
        (
            (1.0, 1.0, 0.1, 0.1, 0.0, 10.0),
            "shot noise; reference phase drift (moving resonance?)",
        ),
    ],
)
def test_diagnosis(args, verdict):
    assert _diagnosis(*args) == verdict


@pytest.mark.parametrize("off", [0.0, np.nan, np.inf])
def test_invalid_reference(off):
    data = stability_data(on=[1.0, 2.0, 3.0], off=[1.0, off, 1.0])
    with np.errstate(divide="raise", invalid="raise"):
        for operation in (lambda: _fit(data), lambda: _plot(data, None, 0)):
            with pytest.raises(ValueError, match="TWPA-off reference magnitude"):
                operation()


@pytest.mark.parametrize(
    "probes, detunings", [(None, None), ([7e9, 7.2e9], None), (None, [-2e6, 0.0])]
)
def test_acquisition(platform, mocker, probes, detunings):
    params = TwpaStabilityParameters.load(
        {
            "repeats": 3,
            "offset": 0.7,
            "nshots": 10,
            "reference_shots": 20,
            "probes": probes,
            "detunings": detunings,
        }
    )
    spy = mocker.spy(platform, "execute")
    data = _acquisition(params, platform, [0, 1])
    nprobes = 1 if probes is None and detunings is None else 2

    # per repetition, a reference and a signal for each probe
    assert spy.call_count == 3 * 2 * nprobes
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    power = platform.config(pump).power
    for index, call in enumerate(spy.call_args_list):
        assert call.kwargs["averaging_mode"] is AveragingMode.SINGLESHOT
        # nothing is swept: the pump plays the point as a continuous wave
        assert len(call.args) == 1
        _, other = call.kwargs["updates"]
        if index % 2 == 0:
            assert other == {pump: {"power": PUMP_OFF_POWER}}
            assert call.kwargs["nshots"] == 20
        else:
            assert other == {
                pump: {
                    "frequency": data.frequency[0],
                    "power": pytest.approx(power + 20 * np.log10(0.7)),
                }
            }
            assert call.kwargs["nshots"] == 10

    assert data.frequency[0] == platform.config(pump).frequency
    assert data.offset == 0.7
    assert data.probes == probes and data.detunings == detunings
    assert (data.shots, data.reference_shots) == (10, 20)
    for qubit in (0, 1):
        assert data[qubit].shape == (3, nprobes, 3)
        assert data.reference_array(qubit).shape == (3, nprobes, 3)
        assert data.nprobes(qubit) == nprobes
    assert len(data.time) == 3 and data.time == sorted(data.time)


def test_acquisition_interval(platform, mocker):
    sleep = mocker.patch("qibocal.protocols.twpa.stability.time.sleep")
    params = TwpaStabilityParameters.load({"repeats": 3, "interval": 2.5, "nshots": 5})
    _acquisition(params, platform, [0])
    # between the repetitions, not before the first one
    assert [call.args for call in sleep.call_args_list] == [(2.5,), (2.5,)]


def test_acquisition_frequency_and_invalid_offset(platform, mocker):
    params = TwpaStabilityParameters.load(
        {"repeats": 2, "frequency": 6.5e9, "nshots": 10}
    )
    spy = mocker.spy(platform, "execute")
    data = _acquisition(params, platform, [0])
    pump = platform.channels[platform.qubits[0].acquisition].twpa_pump
    assert data.frequency == {0: 6.5e9}
    assert spy.call_args_list[1].kwargs["updates"][1][pump]["frequency"] == 6.5e9

    for offset in (1.5, 0.0):
        with pytest.raises(ValueError, match="nonzero, and at most 1"):
            _acquisition(
                TwpaStabilityParameters.load({"repeats": 2, "offset": offset}),
                platform,
                [0],
            )


def test_fit_plot_and_serialization(platform, tmp_path):
    params = TwpaStabilityParameters.load(
        {"repeats": 4, "nshots": 10, "probes": [7e9, 7.2e9]}
    )
    data = _acquisition(params, platform, [0])
    fit = _fit(data)
    assert len(fit.gain[0]) == 2
    for name in TwpaStabilityResults.__dataclass_fields__:
        assert len(getattr(fit, name)[0]) == 2

    figures, report = _plot(data, fit, 0)
    assert len(figures) == 2
    # the gain, with its own and common reference, of each probe
    assert len(figures[0].data) == 4
    # magnitude and phase, on and off, with the shot noise band, of each probe
    assert len(figures[1].data) == 4 * 2 * 2
    for label in (
        "Probe 7.0000 GHz Mean gain",
        "Probe 7.2000 GHz Gain std",
        "On/off",
        "excess over shot noise",
        "Diagnosis",
    ):
        assert label in report
    figures, report = _plot(data, None, 0)
    assert len(figures) == 2 and report == ""

    data.save(tmp_path)
    loaded = TwpaStabilityData.load(tmp_path)
    np.testing.assert_array_equal(data[0], loaded[0])
    np.testing.assert_allclose(data.reference_array(0), loaded.reference_array(0))
    assert loaded.time == data.time and loaded.offset == data.offset
    assert (loaded.shots, loaded.reference_shots) == (data.shots, data.reference_shots)
    fit.save(tmp_path)
    loaded_fit = TwpaStabilityResults.load(tmp_path)
    assert loaded_fit.gain_std == fit.gain_std
    assert loaded_fit.diagnosis == fit.diagnosis
