"""Testing fitting functions"""
import numpy as np
import pytest

from qibocal.data import DataUnits
from qibocal.fitting.methods import (
    drag_tuning_fit,
    flipping_fit,
    lorentzian_fit,
    rabi_fit,
    ramsey_fit,
    res_spectroscopy_flux_fit,
    t1_fit,
)
from qibocal.fitting.utils import (
    cos,
    cumulative,
    exp,
    flipping,
    freq_r_mathieu,
    freq_r_transmon,
    line,
    lorenzian,
    rabi,
    ramsey,
)


@pytest.mark.parametrize("name", [None, "test"])
@pytest.mark.parametrize(
    "qubit, fluxline, num_params",
    [
        (1, 1, 6),
        (1, 1, 7),
        (1, 2, 2),
    ],
)
def test_res_spectroscopy_flux_fit(name, qubit, fluxline, num_params, caplog):
    """Test the *res_spectrocopy_flux_fit* function"""
    x = np.linspace(-0.01, 0.03, 100)
    if num_params == 6:
        p0 = 0.01
        p1 = 41
        p2 = 0.17
        p3 = 0.75
        p4 = 78847979
        p5 = 7651970152
        params = [p0, p1, p2, p3, p4, p5]
        noisy_flux = freq_r_transmon(x, *params) + np.random.randn(100) * 1e-3
        params_fit = [p5, p4]
        labels = [
            "curr_sp",
            "xi",
            "d",
            "f_q/f_rh",
            "g",
            "f_rh",
        ]
    elif num_params == 7:
        p0 = 7651970152
        p1 = 78847979
        p2 = 0.01
        p3 = 41
        p4 = 0.17
        p5 = 2.7e8
        p6 = 1.77e10
        params = [p0, p1, p2, p3, p4, p5, p6]
        noisy_flux = freq_r_mathieu(x, *params) + np.random.randn(100) * 1e-3
        params_fit = [p0, p1, p5, p6]
        labels = [
            "f_rh",
            "g",
            "curr_sp",
            "xi",
            "d",
            "Ec",
            "Ej",
        ]
    else:
        p0 = -4366377
        p1 = 7655179288
        params = [p0, p1]
        noisy_flux = line(x, p0, p1) + np.random.randn(100) * 1e-3
        params_fit = []
        labels = [
            "popt0",
            "popt1",
        ]

    data = DataUnits(quantities={"frequency": "Hz", "current": "A"})

    mydict = {"frequency[Hz]": noisy_flux, "current[A]": x}

    data.load_data_from_dict(mydict)

    fit = res_spectroscopy_flux_fit(
        data, "current[A]", "frequency[Hz]", qubit, fluxline, params_fit
    )

    for j in range(num_params):
        np.testing.assert_allclose(fit.get_values(labels[j])[0], params[j], rtol=0.1)

    x = [0]
    noisy_flux = [0]
    data = DataUnits(quantities={"frequency": "Hz", "current": "A"})
    mydict = {"frequency[Hz]": noisy_flux, "current[A]": x}

    data.load_data_from_dict(mydict)

    fit = res_spectroscopy_flux_fit(
        data, "current[A]", "frequency[Hz]", qubit, fluxline, params_fit
    )
    assert "The fitting was not successful" in caplog.text


@pytest.mark.parametrize("name", [None, "test"])
@pytest.mark.parametrize(
    "label, resonator_type, amplitude_sign",
    [
        ("readout_frequency", "3D", 1),
        ("readout_frequency", "2D", -1),
        ("readout_frequency_shifted", "3D", 1),
        ("readout_frequency_shifted", "2D", -1),
        ("drive_frequency", "3D", -1),
        ("drive_frequency", "2D", 1),
    ],
)
def test_lorentzian_fit(name, label, resonator_type, amplitude_sign, caplog):
    """Test the *lorentzian_fit* function"""
    amplitude = 1 * amplitude_sign
    center = 2
    sigma = 3
    offset = 4

    x = np.linspace(center - 10, center + 10, 100)
    noisy_lorentzian = (
        lorenzian(x, amplitude, center, sigma, offset)
        + amplitude * np.random.randn(100) * 1e-3
    )

    data = DataUnits(quantities={"frequency": "Hz"}, options=["qubit", "iteration"])

    mydict = {
        "frequency[Hz]": x,
        "MSR[V]": noisy_lorentzian,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = lorentzian_fit(
        data,
        "frequency[Hz]",
        "MSR[V]",
        [0],
        resonator_type,
        labels=[label, "peak_voltage"],
        fit_file_name=name,
    )
    # Given the couple (amplitude, sigma) as a solution of lorentzian_fit method
    # also (-amplitude,-sigma) is a possible solution.
    np.testing.assert_allclose(
        abs(fit.get_values("popt0")[0]), abs(amplitude), rtol=0.1
    )
    np.testing.assert_allclose(fit.get_values("popt1")[0], center, rtol=0.1)
    np.testing.assert_allclose(abs(fit.get_values("popt2")[0]), abs(sigma), rtol=0.1)
    np.testing.assert_allclose(fit.get_values("popt3")[0], offset, rtol=0.1)
    np.testing.assert_allclose(fit.get_values(label)[0], 1e9 * center, rtol=0.1)
    # Dummy fit
    x = [0]
    noisy_lorentzian = [0]
    data = DataUnits(quantities={"frequency": "Hz"}, options=["qubit", "iteration"])
    mydict = {
        "frequency[Hz]": x,
        "MSR[V]": noisy_lorentzian,
        "qubit": 0,
        "iteration": 0,
    }

    data.load_data_from_dict(mydict)

    fit = lorentzian_fit(
        data,
        "frequency[Hz]",
        "MSR[V]",
        [0],
        resonator_type,
        labels=[label, "peak_voltage"],
        fit_file_name=name,
    )
    assert "lorentzian_fit: the fitting was not successful" in caplog.text


@pytest.mark.parametrize(
    "label, unit, resonator_type, amplitude_sign",
    [
        ("duration", "ns", "3D", 1),
        ("duration", "ns", "2D", -1),
        ("gain", "dimensionless", "3D", 1),
        ("gain", "dimensionless", "2D", -1),
        ("amplitude", "dimensionless", "3D", 1),
        ("amplitude", "dimensionless", "2D", -1),
    ],
)
def test_rabi_fit(label, unit, resonator_type, amplitude_sign, caplog):
    """Test the *rabi_fit* function"""
    p0 = 4
    p1 = 1 * amplitude_sign
    p2 = 1
    p3 = 2
    p4 = 1 / 5 * 1e-6

    samples = 100
    x = np.linspace(0, 1 / p2, samples)
    noisy_rabi = rabi(x, p0, p1, p2, p3, p4) + p1 * np.random.randn(samples) * 1e-3

    data = DataUnits(quantities={label: unit}, options=["qubit", "iteration"])

    mydict = {
        f"{label}[{unit}]": x,
        "MSR[V]": noisy_rabi,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = rabi_fit(
        data,
        f"{label}[{unit}]",
        "MSR[V]",
        [0],
        resonator_type,
        labels=[f"pi_pulse_{label}", "pi_pulse_peak_voltage"],
    )
    fit_p0 = fit.get_values("popt0")[0]
    fit_p1 = fit.get_values("popt1")[0]
    fit_p2 = fit.get_values("popt2")[0]
    fit_p3 = fit.get_values("popt3")[0]
    fit_p4 = fit.get_values("popt4")[0]
    y_real = p1 * np.sin(2 * np.pi * x * p2 + p3)
    y_fit = fit_p1 * np.sin(2 * np.pi * x * fit_p2 + fit_p3)
    for i in range(len(x)):
        assert abs(y_real[i] - y_fit[i]) < 0.1
    # Dummy fit
    x = [0]
    noisy_rabi = [0]

    data = DataUnits(quantities={label: unit}, options=["qubit", "iteration"])

    mydict = {
        f"{label}[{unit}]": x,
        "MSR[V]": noisy_rabi,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = rabi_fit(
        data,
        f"{label}[{unit}]",
        "MSR[V]",
        [0],
        resonator_type,
        labels=[f"pi_pulse_{label}", "pi_pulse_peak_voltage"],
    )
    assert "rabi_fit: the fitting was not succesful" in caplog.text


@pytest.mark.parametrize(
    "resonator_type, amplitude_sign",
    [
        ("3D", 1),
        ("2D", -1),
    ],
)
def test_ramsey_fit(resonator_type, amplitude_sign, caplog):
    """Test the *ramsey_fit* function"""
    p0 = 4
    p1 = 1 * amplitude_sign
    p2 = 1
    p3 = 2
    p4 = 1 / 5 * 1e-9
    qubit_freqs = [4]
    sampling_rate = 10
    offset_freq = 1
    samples = 100
    x = np.linspace(0, 2 * np.pi / p2, samples)
    noisy_ramsey = ramsey(x, p0, p1, p2, p3, p4) + p1 * np.random.randn(samples) * 1e-3

    data = DataUnits(quantities={"wait": "ns"}, options=["qubit", "iteration"])

    mydict = {
        "wait[ns]": x,
        "MSR[V]": noisy_ramsey,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = ramsey_fit(
        data,
        "wait[ns]",
        "MSR[V]",
        [0],
        resonator_type,
        qubit_freqs,
        sampling_rate,
        offset_freq,
        labels=["delta_frequency", "corrected_qubit_frequency", "t2"],
    )
    fit_p0 = fit.get_values("popt0")[0]
    fit_p1 = fit.get_values("popt1")[0]
    fit_p2 = fit.get_values("popt2")[0]
    fit_p3 = fit.get_values("popt3")[0]
    fit_p4 = fit.get_values("popt4")[0]

    y_real = ramsey(x, p0, p1, p2, p3, p4)
    y_fit = ramsey(x, fit_p0, fit_p1, fit_p2, fit_p3, fit_p4)
    for i in range(len(x)):
        assert abs(y_real[i] - y_fit[i]) < 0.1
    # Dummy fit
    x = [0]
    noisy_ramsey = [0]

    data = DataUnits(quantities={"wait": "ns"}, options=["qubit", "iteration"])

    mydict = {
        "wait[ns]": x,
        "MSR[V]": noisy_ramsey,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = ramsey_fit(
        data,
        "wait[ns]",
        "MSR[V]",
        [0],
        resonator_type,
        qubit_freqs,
        sampling_rate,
        offset_freq,
        labels=["delta_frequency", "corrected_qubit_frequency", "t2"],
    )
    assert "ramsey_fit: the fitting was not succesful" in caplog.text


@pytest.mark.parametrize(
    "label, resonator_type, amplitude_sign",
    [
        ("t1", "3D", 1),
        ("t1", "2D", -1),
    ],
)
def test_t1_fit(label, resonator_type, amplitude_sign, caplog):
    """Test the *t1_fit* function"""
    p0 = 0
    p1 = 1 * amplitude_sign
    p2 = 1 / 5

    x = np.linspace(0, 10, 100)
    noisy_t1 = exp(x, p0, p1, p2) + np.random.randn(100) * 1e-4

    data = DataUnits(quantities={"wait": "ns"}, options=["qubit", "iteration"])

    mydict = {
        "wait[ns]": x,
        "MSR[V]": noisy_t1,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = t1_fit(data, "wait[ns]", "MSR[V]", [0], resonator_type, labels=[label])

    fit_p = [fit.get_values(f"popt{i}")[0] for i in range(3)]
    fit_t1 = exp(x, *fit_p)
    for i in range(len(x)):
        assert abs(fit_t1[i] - noisy_t1[i]) < 0.2
    # Dummy fit
    x = [0]
    noisy_t1 = [0]
    data = DataUnits(quantities={"wait": "ns"}, options=["qubit", "iteration"])
    mydict = {
        "wait[ns]": x,
        "MSR[V]": noisy_t1,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = t1_fit(data, "wait[ns]", "MSR[V]", [0], resonator_type, labels=[label])
    assert "t1_fit: the fitting was not succesful" in caplog.text


@pytest.mark.parametrize(
    "label, resonator_type, amplitude_sign",
    [
        ("amplitude_correction_factor", "3D", 1),
        ("amplitude_correction_factor", "2D", -1),
    ],
)
def test_flipping_fit(label, resonator_type, amplitude_sign, caplog):
    """Test the *flipping_fit* function"""
    p0 = 0.0001 * amplitude_sign
    p1 = 1
    p2 = 17 * amplitude_sign
    p3 = 3

    pi_pulse_amplitudes = [5]

    x = np.linspace(0, 10, 100)
    noisy_flip = flipping(x, p0, p1, p2, p3) + p0 * np.random.randn(100) * 1e-4

    data = DataUnits(
        quantities={"flips": "dimensionless"}, options=["qubit", "iteration"]
    )

    mydict = {
        "flips[dimensionless]": x,
        "MSR[V]": noisy_flip,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = flipping_fit(
        data,
        "flips[dimensionless]",
        "MSR[V]",
        [0],
        resonator_type,
        pi_pulse_amplitudes,
        labels=[label, "corrected_amplitude"],
    )
    fit_p = [fit.get_values(f"popt{i}")[0] for i in range(4)]
    fit_flip = flipping(x, *fit_p)
    for i in range(len(x)):
        assert abs(fit_flip[i] - noisy_flip[i]) < 0.2
    # Dummy fit
    x = [0, 0]
    noisy_flip = [0, 0]
    data = DataUnits(
        quantities={"flips": "dimensionless"}, options=["qubit", "iteration"]
    )

    mydict = {
        "flips[dimensionless]": x,
        "MSR[V]": noisy_flip,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = flipping_fit(
        data,
        "flips[dimensionless]",
        "MSR[V]",
        [0],
        resonator_type,
        pi_pulse_amplitudes,
        labels=[label, "corrected_amplitude"],
    )
    assert "flipping_fit: the fitting was not succesful" in caplog.text


@pytest.mark.parametrize(
    "label",
    [
        ("optimal_beta_param"),
    ],
)
def test_drag_tunning_fit(label, caplog):
    """Test the *drag_tunning_fit* function"""
    p0 = 0
    p1 = 0.1
    p2 = 4
    p3 = 0.3

    x = np.linspace(0, 10, 100)
    noisy_drag = cos(x, p0, p1, p2, p3) + p1 * np.random.randn(100) * 1e-6

    data = DataUnits(
        quantities={"beta_param": "dimensionless"}, options=["qubit", "iteration"]
    )

    mydict = {
        "beta_param[dimensionless]": x,
        "MSR[V]": noisy_drag,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = drag_tuning_fit(
        data, "beta_param[dimensionless]", "MSR[V]", [0], labels=label
    )
    fit_p = [fit.get_values(f"popt{i}")[0] for i in range(4)]
    fit_drag = flipping(x, *fit_p)
    MSQE = 0
    for i in range(len(x)):
        MSQE += abs(fit_drag[i] - noisy_drag[i])
    assert MSQE / len(x) < 0.1
    # Dummy fit
    x = [0, 0]
    noisy_drag = [0, 0]
    data = DataUnits(
        quantities={"beta_param": "dimensionless"}, options=["qubit", "iteration"]
    )

    mydict = {
        "beta_param[dimensionless]": x,
        "MSR[V]": noisy_drag,
        "qubit": [0] * len(x),
        "iteration": [0] * len(x),
    }

    data.load_data_from_dict(mydict)

    fit = drag_tuning_fit(
        data, "beta_param[dimensionless]", "MSR[V]", [0], labels=label
    )
    assert "drag_tuning_fit: the fitting was not succesful" in caplog.text


def test_cumulative():
    points = x = np.linspace(0, 9, 10)
    cum = cumulative(x, points)
    assert np.array_equal(cum, points)
