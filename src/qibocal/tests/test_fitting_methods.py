# -*- coding: utf-8 -*-
import numpy as np
import pytest

from qibocal.data import DataUnits
from qibocal.fitting.methods import lorentzian_fit
from qibocal.fitting.utils import lorenzian


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
def test_lorentzian_fit(name, label, nqubits, amplitude_sign):

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

    np.testing.assert_allclose(fit.get_values("popt0")[0], amplitude, rtol=0.1)
    np.testing.assert_allclose(fit.get_values("popt1")[0], center, rtol=0.1)
    np.testing.assert_allclose(fit.get_values("popt2")[0], sigma, rtol=0.1)
    np.testing.assert_allclose(fit.get_values("popt3")[0], offset, rtol=0.1)
    np.testing.assert_allclose(fit.get_values(label)[0], center * 1e9, rtol=0.1)
