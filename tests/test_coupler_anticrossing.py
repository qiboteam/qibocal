import inspect

import numpy as np
import pytest
from qibolab import Parameter

from qibocal import update
from qibocal.calibration.calibration import TwoQubitCalibration
from qibocal.protocols.couplers import anti_crossing as ac
from qibocal.protocols.couplers import anti_crossing_flux as acf
from qibocal.protocols.flux_dependence.qubit_flux_dependence import (
    _extract_peak_coordinates,
)

PAIR = (0, 1)
COUPLER = 1
PARABOLA = [-3e9, -3.6e8, 5.5892e9]
"""Coupler frequency [Hz] versus bias: vertex at -0.06, 5.6 GHz."""
G = 10e6
BIAS = np.arange(-0.6, 0.6, 0.003)
QUBITS = {0: 5.0e9, 1: 5.3e9}


def coupler(x):
    return np.polyval(PARABOLA, x)


def true_crossings(frequency):
    return sorted(np.roots(np.subtract(PARABOLA, [0, 0, frequency])).real)


def spectrum(fq, frequency, noise=0.0):
    """Magnitude of both hybridised branches, weighted by their qubit content."""
    fc = coupler(BIAS)[:, None]
    half = np.hypot((fc - fq) / 2, G)
    signal = sum(
        0.5
        * (1 - b * (fc - fq) / 2 / half)
        / (1 + ((frequency - (fc + fq) / 2 - b * half) / 1e6) ** 2)
        for b in (1, -1)
    )
    rng = np.random.default_rng(0)
    noise = noise * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
    return np.abs(1 + 0.5 * signal + noise)


def record(fq, noise=0.0):
    frequency = np.arange(fq - 30e6, fq + 30e6, 0.5e6)
    bias, freq = np.meshgrid(BIAS, frequency, indexing="ij")
    signal = spectrum(fq, frequency, noise)
    return {"bias": bias.ravel(), "frequency": freq.ravel(), "signal": signal.ravel()}


def pair_data(noise=0.0):
    data = ac.CouplerAnticrossingData()
    for qubit, fq in QUBITS.items():
        data.register_qubit(
            ac.CouplerAnticrossingType, (*PAIR, qubit), record(fq, noise)
        )
    return data


def flux_data(qubit_biases=(0.0, 0.05, 0.1, 0.15, 0.2)):
    data = acf.CouplerAnticrossingFluxData()
    for value in qubit_biases:
        spectrum_ = record(5.3e9 - 1e10 * value**2)
        spectrum_["qubit_bias"] = np.full(len(spectrum_["bias"]), value)
        data.register_qubit(acf.CouplerAnticrossingFluxType, PAIR, spectrum_)
    return data


def check_parabola(coupler):
    assert coupler is not None
    for value, fitted, error in zip(
        PARABOLA, coupler["coefficients"], coupler["errors"]
    ):
        assert abs(fitted - value) <= 3 * error
    assert coupler["sweetspot"] == pytest.approx(-0.06, abs=1e-3)
    assert coupler["frequency"] == pytest.approx(5.6e9, abs=3e6)


def check_crossings(crossings):
    for c in crossings:
        fq = coupler(c["bias"])
        expected = min(true_crossings(c["frequency"]), key=lambda x: abs(x - c["bias"]))
        assert abs(c["bias"] - expected) <= 3 * c["bias_error"]
        assert np.sign(c["fitted_slope"]) == np.sign(
            np.polyval(np.polyder(PARABOLA), expected)
        )
        assert c["frequency"] == pytest.approx(fq, abs=3e6)


@pytest.fixture(scope="module")
def pair_fit():
    data = pair_data()
    return data, ac._fit(data)


def test_extractor_signature_unchanged():
    parameters = inspect.signature(_extract_peak_coordinates).parameters
    assert list(parameters) == ["frequencies", "biases", "signal"]


@pytest.mark.parametrize("sc", [3e9, -3e9])
def test_local_fit_on_exact_branch(sc):
    """Bare slope (twice the traced tangent at the crossing), point and gap."""
    bias = np.arange(-0.05, 0.05, 0.001) + 1e-4
    xc, fq = 0.0013, 5e9
    qubit_like = -np.sign(sc * (bias - xc))
    trace = np.round(ac.branch(bias, xc, fq, sc, G, qubit_like) / 0.5e6) * 0.5e6
    frequency = np.arange(4.96e9, 5.04e9, 0.5e6)
    signal = (np.abs(frequency - trace[:, None]) < 1).astype(float)
    (c,) = ac.crossings(bias, frequency, signal)
    assert abs(c["bias"] - xc) <= 3 * c["bias_error"]
    assert abs(c["frequency"] - fq) <= 3 * c["frequency_error"]
    assert c["gap"] == pytest.approx(2 * G, rel=0.1)
    assert c["slope"] is not None
    assert abs(c["slope"] - sc) <= 3 * c["slope_error"]


def test_parabola_from_points_and_tangents():
    derivative = np.polyder(PARABOLA)
    constraints = [
        {
            "bias": x,
            "bias_error": 1e-4,
            "frequency": coupler(x),
            "frequency_error": 0.2e6,
            "fitted_slope": np.polyval(derivative, x),
            "slope": np.polyval(derivative, x) if tangent else None,
            "slope_error": 0.03 * abs(np.polyval(derivative, x)),
        }
        for x, tangent in ((-0.5, True), (0.4, True), (-0.3, False))
    ]
    check_parabola(ac.parabola_fit(constraints))
    # Two points, or a point and a tangent, do not fix three coefficients.
    assert ac.parabola_fit([constraints[2], constraints[2] | {"bias": 0.1}]) is None
    assert ac.parabola_fit(constraints[:1]) is None


def test_pair_crossings_and_parabola(pair_fit):
    _, results = pair_fit
    crossings = results.crossings[PAIR]
    assert sorted(c["qubit"] for c in crossings) == [0, 0, 1, 1]
    check_crossings(crossings)
    check_parabola(results.coupler[PAIR])


@pytest.mark.xfail(
    strict=True,
    reason="the shared qubit_flux extractor reports the bare qubit bin inside each"
    " step, so the bend is not traced and no tangent is measured",
)
def test_pair_tangents(pair_fit):
    assert all(c["slope"] is not None for c in pair_fit[1].crossings[PAIR])


def test_noisy_map_gives_no_crossings():
    """The ZCA whitening of the shared extractor scatters a noisy flat line: no
    crossings are fitted from such traces."""
    results = ac._fit(pair_data(noise=0.03))
    assert results.crossings[PAIR] == [] and PAIR not in results.coupler


def test_flux_variant_uses_all_qubit_biases():
    results = acf._fit(flux_data())
    crossings = results.crossings[PAIR]
    assert len(crossings) == 10
    assert len({c["qubit_bias"] for c in crossings}) == 5
    check_crossings(crossings)
    check_parabola(results.coupler[PAIR])


def test_serialization_and_plots(pair_fit, tmp_path):
    data, results = pair_fit
    data.save(tmp_path)
    results.save(tmp_path)
    loaded = ac.CouplerAnticrossingResults.load(tmp_path)
    assert loaded.crossings == results.crossings
    assert loaded.coupler == results.coupler
    figures, report = ac._plot(ac.CouplerAnticrossingData.load(tmp_path), PAIR, loaded)
    names = [trace.name for trace in figures[0].data]
    assert sum(name.startswith("Crossing") for name in names) == 4
    assert "Coupler parabola" in names and "Sweetspot" in report
    assert ac._plot(data, PAIR) == ([], "")
    flux = flux_data([0.0, 0.1])
    figures, _ = acf._plot(flux, PAIR, acf._fit(flux))
    assert "Coupler parabola" in [trace.name for trace in figures[0].data]


@pytest.fixture
def coupler_platform(platform):
    platform.calibration.two_qubits[PAIR] = TwoQubitCalibration(coupler=COUPLER)
    update.flux_offset(0.25, platform, COUPLER, coupler=True)
    return platform


def mock_execute(platform, monkeypatch):
    calls = []

    def execute(sequences, sweepers, **kwargs):
        calls.append((sequences, sweepers, kwargs))
        shape = tuple(len(s[0].values) for s in sweepers) + (2,)
        readout = [p for _, p in sequences[0] if type(p).__name__ == "Readout"]
        return {readout[0].id: np.ones(shape)}

    monkeypatch.setattr(platform, "execute", execute)
    return calls


COMMON = {
    "bias": ["linspace", -0.2, 0.2, 11],
    "pulse_duration": 1000,
    "pulse_amplitude": 0.1,
    "nshots": 10,
    "relaxation_time": 100,
}


def test_acquisition_one_execution_per_qubit(coupler_platform, monkeypatch):
    platform = coupler_platform
    calls = mock_execute(platform, monkeypatch)
    params = ac.CouplerAnticrossingParameters.load(
        COMMON
        | {"frequency_0": ["center", 10e6, 1e6], "frequency_1": ["center", 20e6, 1e6]}
    )
    data = ac._acquisition(params, platform, [PAIR])
    assert len(calls) == 2
    for qubit, width, (sequences, sweepers, kwargs) in zip(PAIR, (10e6, 20e6), calls):
        natives = platform.natives.single_qubit[qubit]
        drive = natives.RX()[0][0]
        channels = {channel for channel, _ in sequences[0]}
        assert channels == {drive, natives.MZ()[0][0], platform.couplers[COUPLER].flux}
        bias_sweeper, frequency_sweeper = (s[0] for s in sweepers)
        assert bias_sweeper.parameter is Parameter.amplitude
        assert frequency_sweeper.channels == [drive]
        center = platform.config(drive).frequency
        assert np.ptp(frequency_sweeper.values) == pytest.approx(width - 1e6)
        assert np.mean(frequency_sweeper.values) == pytest.approx(center - 0.5e6)
        assert list(kwargs["updates"][0]) == [platform.qubits[qubit].probe]
    assert set(data.data) == {(*PAIR, 0), (*PAIR, 1)}


def test_flux_acquisition_one_execution_per_qubit_bias(coupler_platform, monkeypatch):
    platform = coupler_platform
    calls = mock_execute(platform, monkeypatch)
    flux = platform.qubits[0].flux
    offset = platform.config(flux).offset
    params = acf.CouplerAnticrossingFluxParameters.load(
        COMMON | {"qubit_bias": [0.0, 0.03, 0.01], "frequency": ["center", 10e6, 1e6]}
    )
    data = acf._acquisition(params, platform, [PAIR])
    offsets = [kwargs["updates"][0][flux]["offset"] for _, _, kwargs in calls]
    assert offsets == pytest.approx(offset + np.array([0.0, 0.01, 0.02]))
    drives = {sweepers[1][0].channels[0] for _, sweepers, _ in calls}
    assert drives == {platform.natives.single_qubit[0].RX()[0][0]}
    assert np.unique(data[PAIR]["qubit_bias"]) == pytest.approx([0.0, 0.01, 0.02])


def test_update(coupler_platform, pair_fit):
    platform = coupler_platform
    results = pair_fit[1]
    ac._update(results, platform, PAIR)
    calibration = platform.calibration.couplers[COUPLER].qubit
    assert calibration.sweetspot == pytest.approx(
        0.25 + results.coupler[PAIR]["sweetspot"]
    )
    assert calibration.maximum_frequency == results.coupler[PAIR]["frequency"]
    # A parabola without maximum is not a transmon coupler: nothing is updated.
    results = ac.CouplerAnticrossingResults(
        coupler={PAIR: results.coupler[PAIR] | {"coefficients": [1.0, 0.0, 0.0]}}
    )
    ac._update(results, platform, PAIR)
    assert calibration.maximum_frequency == pair_fit[1].coupler[PAIR]["frequency"]
