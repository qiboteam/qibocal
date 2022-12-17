import logging

import numpy as np
import pytest

from qibocal.config import log
from qibocal.data import DataUnits
from qibocal.fitting.methods import res_spectrocopy_flux_fit
from qibocal.fitting.utils import freq_r_mathieu, freq_r_transmon, line


@pytest.mark.parametrize("name", [None, "test"])
@pytest.mark.parametrize(
    "qubit, fluxline, num_params",
    [
        (1, 1, 6),
        (1, 1, 7),
        (1, 2, 2),
    ],
)
def test_res_spectrocopy_flux_fit(name, qubit, fluxline, num_params, caplog):
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

    fit = res_spectrocopy_flux_fit(
        data, "current[A]", "frequency[Hz]", qubit, fluxline, params_fit
    )

    for j in range(num_params):
        np.testing.assert_allclose(fit.get_values(labels[j])[0], params[j], rtol=0.1)

    x = [0]
    noisy_flux = [0]
    data = DataUnits(quantities={"frequency": "Hz", "current": "A"})
    mydict = {"frequency[Hz]": noisy_flux, "current[A]": x}

    data.load_data_from_dict(mydict)

    fit = res_spectrocopy_flux_fit(
        data, "current[A]", "frequency[Hz]", qubit, fluxline, params_fit
    )
    assert "The fitting was not successful" in caplog.text