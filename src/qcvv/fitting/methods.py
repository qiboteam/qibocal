# -*- coding: utf-8 -*-
"""Routine-specific method for post-processing data acquired."""
import lmfit
import numpy as np

from qcvv.config import log
from qcvv.data import Data
from qcvv.fitting.utils import lorenzian, parse


def lorentzian_fit(data, x, y, qubit, nqubits, labels):
    """Fitting routine for resonator spectroscopy"""

    data_fit = Data(
        name=f"fit_q{qubit}",
        quantities=[
            "fit_amplitude",
            "fit_center",
            "fit_sigma",
            "fit_offset",
            labels[1],
            labels[0],
        ],
    )

    frequencies = data.get_values(*parse(x))
    voltages = data.get_values(*parse(y))

    # Create a lmfit model for fitting equation defined in resonator_peak
    model_Q = lmfit.Model(lorenzian)

    # Guess parameters for Lorentzian max or min
    if nqubits == 1 and labels[0] == "resonator_freq":
        guess_center = frequencies[
            np.argmax(voltages)
        ]  # Argmax = Returns the indices of the maximum values along an axis.
        guess_offset = np.mean(
            voltages[np.abs(voltages - np.mean(voltages) < np.std(voltages))]
        )
        guess_sigma = abs(frequencies[np.argmin(voltages)] - guess_center)
        guess_amp = (np.max(voltages) - guess_offset) * guess_sigma * np.pi

    else:
        guess_center = frequencies[
            np.argmin(voltages)
        ]  # Argmin = Returns the indices of the minimum values along an axis.
        guess_offset = np.mean(
            voltages[np.abs(voltages - np.mean(voltages) < np.std(voltages))]
        )
        guess_sigma = abs(frequencies[np.argmax(voltages)] - guess_center)
        guess_amp = (np.min(voltages) - guess_offset) * guess_sigma * np.pi

    # Add guessed parameters to the model
    model_Q.set_param_hint("center", value=guess_center, vary=True)
    model_Q.set_param_hint("sigma", value=guess_sigma, vary=True)
    model_Q.set_param_hint("amplitude", value=guess_amp, vary=True)
    model_Q.set_param_hint("offset", value=guess_offset, vary=True)
    guess_parameters = model_Q.make_params()

    # fit the model with the data and guessed parameters
    try:
        fit_res = model_Q.fit(
            data=voltages, frequency=frequencies, params=guess_parameters
        )
    except:
        log.warning("The fitting was not successful")
        return data_fit

    # get the values for postprocessing and for legend.
    f0 = fit_res.best_values["center"]
    BW = fit_res.best_values["sigma"] * 2
    Q = abs(f0 / BW)
    peak_voltage = (
        fit_res.best_values["amplitude"] / (fit_res.best_values["sigma"] * np.pi)
        + fit_res.best_values["offset"]
    )

    freq = f0 * 1e6

    data_fit.add(
        {
            labels[1]: peak_voltage,
            labels[0]: freq,
            "fit_amplitude": fit_res.best_values["amplitude"],
            "fit_center": fit_res.best_values["center"],
            "fit_sigma": fit_res.best_values["sigma"],
            "fit_offset": fit_res.best_values["offset"],
        }
    )
    return data_fit

    # params = resonator_freq, peak_voltage

    # for keys in fit_res.best_values:
    #     fit_res.best_values[keys] = float(fit_res.best_values[keys])

    # with open(f"{folder}/data/resonator_spectroscopy/fit.yml", "w+") as file:
    #     yaml.dump(
    #         fit_res.best_values,
    #         file,
    #         sort_keys=False,
    #         indent=4,
    #         default_flow_style=None,
    #     )

    # return params, fit_res.best_values
