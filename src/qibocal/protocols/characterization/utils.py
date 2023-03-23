import lmfit
import numpy as np

from ...auto.operation import Results
from ...config import log
from ...data import DataUnits


def lorentzian(frequency, amplitude, center, sigma, offset):
    # http://openafox.com/science/peak-function-derivations.html
    return (amplitude / np.pi) * (
        sigma / ((frequency - center) ** 2 + sigma**2)
    ) + offset


def lorentzian_fit(data: DataUnits) -> list:
    qubits = data.df["qubit"].unique()
    resonator_type = data.df["resonator_type"].unique()
    amplitudes = {}
    frequency = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration", "resonator_type", "amplitude"])
            .groupby("frequency", as_index=False)
            .mean()
        )

        frequencies = qubit_data["frequency"].pint.to("GHz").pint.magnitude

        voltages = qubit_data["MSR"].pint.to("uV").pint.magnitude

        model_Q = lmfit.Model(lorentzian)

        if resonator_type == "3D":
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
            # get the values for postprocessing and for legend.
            f0 = fit_res.best_values["center"]
            BW = fit_res.best_values["sigma"] * 2
            Q = abs(f0 / BW)
            peak_voltage = (
                fit_res.best_values["amplitude"]
                / (fit_res.best_values["sigma"] * np.pi)
                + fit_res.best_values["offset"]
            )
            freq = f0

        except:
            log.warning("lorentzian_fit: the fitting was not successful")

        frequency[qubit] = f0
        data_df = data.df
        amplitude = data_df[data_df.qubit == qubit]["amplitude"].unique()
        amplitudes[qubit] = amplitude[0]
        fitted_parameters[qubit] = fit_res.best_values

    return [frequency, amplitudes, fitted_parameters]
