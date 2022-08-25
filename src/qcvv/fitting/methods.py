# -*- coding: utf-8 -*-
"""Routine-specific method for post-processing data acquired."""
import lmfit
import numpy as np

from qcvv.data import Dataset
from qcvv.fitting.utils import get_values, lorenzian


def resonator_spectroscopy_fit(folder, format, platform, qubit, params):
    """Fitting routine for resonator spectroscopy"""

    data_fast = Dataset.load_data(
        folder, "resonator_spectroscopy", format, "fast_sweep"
    )

    lowres_width = params["lowres_width"]
    lowres_step = params["lowres_step"]
    nqubits = platform.settings["nqubits"]

    if nqubits == 1:
        avg_voltage = (
            np.mean(data_fast.df.MSR.values[: (lowres_width // lowres_step)]) * 1e6
        )
    else:
        avg_voltage = (
            np.mean(data_fast.df.MSR.values[: (lowres_width // lowres_step)]) * 1e6
        )

    data = Dataset.load_data(
        folder, "resonator_spectroscopy", format, "precision_sweep"
    )

    voltages = get_values(data.df, "MSR", "V")
    frequencies = get_values(data.df, "frequency", "Hz")

    # Create a lmfit model for fitting equation defined in resonator_peak
    model_Q = lmfit.Model(lorenzian)

    # Guess parameters for Lorentzian max or min
    if nqubits == 1:
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
    fit_res = model_Q.fit(data=voltages, frequency=frequencies, params=guess_parameters)

    # get the values for postprocessing and for legend.
    f0 = fit_res.best_values["center"]
    BW = fit_res.best_values["sigma"] * 2
    Q = abs(f0 / BW)
    peak_voltage = (
        fit_res.best_values["amplitude"] / (fit_res.best_values["sigma"] * np.pi)
        + fit_res.best_values["offset"]
    )
    peak_voltage *= 1e6

    resonator_freq = f0  # + platform.qubit_readout_pulse(qubit, start=0).frequency

    return resonator_freq, avg_voltage, peak_voltage
