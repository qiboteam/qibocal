# -*- coding: utf-8 -*-
import logging

import numpy as np
import pytest

from qibocal.config import log
from qibocal.data import DataUnits
from qibocal.fitting.methods import (
    drag_tunning_fit,
    flipping_fit,
    lorentzian_fit,
    rabi_fit,
    ramsey_fit,
    t1_fit,
)
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


@pytest.mark.parametrize("name", [None, "test"])
@pytest.mark.parametrize(
    "label, nqubits, amplitude_sign",
    [
        ("resonator_freq", 1, 1),
        ("resonator_freq", 5, -1),
        ("qubit_freq", 1, -1),
        ("qubit_freq", 5, 1),
    ],
)
def test_lorentzian_fit(name, label, nqubits, amplitude_sign, caplog):
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

    data = DataUnits(quantities={"frequency": "Hz"})

    mydict = {"frequency[Hz]": x, "MSR[V]": noisy_lorentzian}

    data.load_data_from_dict(mydict)

    fit = lorentzian_fit(
        data,
        "frequency[Hz]",
        "MSR[V]",
        0,
        nqubits,
        labels=[label, "peak_voltage"],
        fit_file_name=name,
    )
    # Given the couople (amplitude, sigma) as a solution of lorentzian_fit method
    # also (-amplitude,-sigma) is a possible solution.

    np.testing.assert_allclose(
        abs(fit.get_values("popt0")[0]), abs(amplitude), rtol=0.1
    )
    np.testing.assert_allclose(fit.get_values("popt1")[0], center, rtol=0.1)
    np.testing.assert_allclose(abs(fit.get_values("popt2")[0]), abs(sigma), rtol=0.1)
    np.testing.assert_allclose(fit.get_values("popt3")[0], offset, rtol=0.1)
    np.testing.assert_allclose(fit.get_values(label)[0], center * 1e9, rtol=0.1)
    # Dummy fit
    x = [0]
    noisy_lorentzian = [0]
    data = DataUnits(quantities={"frequency": "Hz"})
    mydict = {"frequency[Hz]": x, "MSR[V]": noisy_lorentzian}

    data.load_data_from_dict(mydict)

    fit = lorentzian_fit(
        data,
        "frequency[Hz]",
        "MSR[V]",
        0,
        nqubits,
        labels=[label, "peak_voltage"],
        fit_file_name=name,
    )
    assert "The fitting was not successful" in caplog.text


@pytest.mark.parametrize(
    "label, nqubits, amplitude_sign",
    [
        ("pulse_duration", 1, 1),
        ("pulse_duration", 5, 1),
        ("pulse_duration", 1, -1),
        ("pulse_duration", 5, -1),
    ],
)
def test_rabi_fit(label, nqubits, amplitude_sign, caplog):
    """Test the *rabi_fit* function"""
    p0 = 4
    p1 = 1 * amplitude_sign
    p2 = 1
    p3 = 2
    p4 = 1 / 5 * 1e-6

    samples = 100
    x = np.linspace(0, 1 / p2, samples)
    noisy_rabi = rabi(x, p0, p1, p2, p3, p4) + p1 * np.random.randn(samples) * 1e-3

    data = DataUnits(quantities={"time": "s"})

    mydict = {"time[s]": x, "MSR[V]": noisy_rabi}

    data.load_data_from_dict(mydict)

    fit = rabi_fit(
        data, "time[s]", "MSR[V]", 0, nqubits, labels=[label, "pulse_max_voltage"]
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

    data = DataUnits(quantities={"time": "s"})

    mydict = {"time[s]": x, "MSR[V]": noisy_rabi}

    data.load_data_from_dict(mydict)

    fit = rabi_fit(
        data, "time[s]", "MSR[V]", 0, nqubits, labels=[label, "pulse_max_voltage"]
    )
    assert "The fitting was not succesful" in caplog.text


@pytest.mark.parametrize("amplitude_sign", [-1, 1])
def test_ramsey_fit(amplitude_sign, caplog):
    """Test the *ramsey_fit* function"""
    p0 = 4
    p1 = 1 * amplitude_sign
    p2 = 1
    p3 = 2
    p4 = 1 / 5 * 1e-9
    qubit_freq = 4
    sampling_rate = 10
    offset_freq = 1
    samples = 100
    x = np.linspace(0, 1 / p2, samples)
    noisy_ramsey = ramsey(x, p0, p1, p2, p3, p4) + p1 * np.random.randn(samples) * 1e-3

    data = DataUnits(quantities={"time": "s"})

    mydict = {"time[s]": x, "MSR[V]": noisy_ramsey}

    data.load_data_from_dict(mydict)

    fit = ramsey_fit(
        data,
        "time[s]",
        "MSR[V]",
        0,
        qubit_freq,
        sampling_rate,
        offset_freq,
        labels=["delta_phys", "qubit_freq", "t2"],
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
    noisy_ramsey = [0]

    data = DataUnits(quantities={"time": "s"})

    mydict = {"time[s]": x, "MSR[V]": noisy_ramsey}

    data.load_data_from_dict(mydict)

    fit = ramsey_fit(
        data,
        "time[s]",
        "MSR[V]",
        0,
        qubit_freq,
        sampling_rate,
        offset_freq,
        labels=["delta_phys", "qubit_freq", "t2"],
    )
    assert "The fitting was not succesful" in caplog.text


@pytest.mark.parametrize(
    "label, nqubits, amplitude_sign",
    [
        ("resonator_freq", 1, 1),
        ("resonator_freq", 5, -1),
        ("qubit_freq", 1, -1),
        ("qubit_freq", 5, 1),
    ],
)
def test_t1_fit(label, nqubits, amplitude_sign, caplog):
    """Test the *t1_fit* function"""
    p0 = 0
    p1 = 1 * amplitude_sign
    p2 = 1 / 5

    x = np.linspace(0, 10, 100)
    noisy_t1 = exp(x, p0, p1, p2) + np.random.randn(100) * 1e-4

    data = DataUnits(quantities={"time": "s"})

    mydict = {"time[s]": x, "MSR[V]": noisy_t1}

    data.load_data_from_dict(mydict)

    fit = t1_fit(data, "time[s]", "MSR[V]", 0, nqubits, labels=[label])

    fit_p = [fit.get_values(f"popt{i}")[0] for i in range(3)]
    fit_t1 = exp(x, *fit_p)
    for i in range(len(x)):
        assert abs(fit_t1[i] - noisy_t1[i]) < 0.2
    # Dummy fit
    x = [0]
    noisy_t1 = [0]
    data = DataUnits(quantities={"time": "s"})
    mydict = {"time[s]": x, "MSR[V]": noisy_t1}

    data.load_data_from_dict(mydict)

    fit = t1_fit(data, "time[s]", "MSR[V]", 0, nqubits, labels=[label])
    assert "The fitting was not succesful" in caplog.text


@pytest.mark.parametrize(
    "label, nqubits, amplitude_sign",
    [
        ("resonator_ampl", 1, 1),
        ("resonator_amp", 5, -1),
        ("qubit_ampl", 1, -1),
        ("qubit_ampl", 5, 1),
    ],
)
def test_flipping_fit(label, nqubits, amplitude_sign, caplog):
    """Test the *flipping_fit* function"""
    p0 = 0.0001 * amplitude_sign
    p1 = 1
    p2 = 17 * amplitude_sign
    p3 = 3

    pi_pulse_amplituse = 5

    x = np.linspace(0, 10, 100)
    noisy_flip = flipping(x, p0, p1, p2, p3) + p0 * np.random.randn(100) * 1e-4

    data = DataUnits(quantities={"flips": "N"})

    mydict = {"flips[N]": x, "MSR[V]": noisy_flip}

    data.load_data_from_dict(mydict)

    fit = flipping_fit(
        data,
        "flips[N]",
        "MSR[V]",
        0,
        nqubits,
        0,
        pi_pulse_amplituse,
        labels=[label, "corrected_amplitude"],
    )
    fit_p = [fit.get_values(f"popt{i}")[0] for i in range(4)]
    fit_flip = flipping(x, *fit_p)
    for i in range(len(x)):
        assert abs(fit_flip[i] - noisy_flip[i]) < 0.2
    # Dummy fit
    x = [0, 0]
    noisy_flip = [0, 0]
    data = DataUnits(quantities={"flips": "N"})

    mydict = {"flips[N]": x, "MSR[V]": noisy_flip}

    data.load_data_from_dict(mydict)

    fit = flipping_fit(
        data,
        "flips[N]",
        "MSR[V]",
        0,
        nqubits,
        0,
        pi_pulse_amplituse,
        labels=[label, "corrected_amplitude"],
    )
    assert "The fitting was not succesful" in caplog.text


@pytest.mark.parametrize(
    "label",
    [
        ("resonator_ampl"),
        ("resonator_amp"),
        ("qubit_ampl"),
        ("qubit_ampl"),
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

    data = DataUnits(quantities={"beta": "N"})

    mydict = {"beta[N]": x, "MSR[V]": noisy_drag}

    data.load_data_from_dict(mydict)

    fit = drag_tunning_fit(data, "beta[N]", "MSR[V]", 0, 1, labels=label)
    fit_p = [fit.get_values(f"popt{i}")[0] for i in range(4)]
    fit_drag = flipping(x, *fit_p)
    MSQE = 0
    for i in range(len(x)):
        MSQE += abs(fit_drag[i] - noisy_drag[i])
    assert MSQE / len(x) < 0.1
    # Dummy fit
    x = [0, 0]
    noisy_drag = [0, 0]
    data = DataUnits(quantities={"beta": "N"})

    mydict = {"beta[N]": x, "MSR[V]": noisy_drag}

    data.load_data_from_dict(mydict)

    fit = drag_tunning_fit(data, "beta[N]", "MSR[V]", 0, 1, labels=label)
    assert "The fitting was not succesful" in caplog.text
