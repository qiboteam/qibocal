from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from qibo.backends import GlobalBackend

from qibocal.auto.operation import Data, Parameters, Results
from qibocal.auto.task import Routine
from qibocal.fitting.methods import lorentzian_fit, lorenzian

from .hardware.resonator_spectroscopy import resonator_spectroscopy as test


@dataclass
class ResonatorSpectroscopyParameters(Parameters):
    freq_width: int
    freq_step: int


@dataclass
class ResonatorSpectroscopyData(Data):
    frequency: npt.NDArray[np.float64]
    phase: npt.NDArray[np.float64]
    msr: npt.NDArray[np.float64]
    qubit: npt.NDArray[np.float64]


@dataclass
class ResonatorSpectroscoyResults(Results):
    """ResonatorSpectroscopyResults"""


def _resonator_spectroscopy_acquisition(
    platform, qubits, args: ResonatorSpectroscopyParameters
):
    data = test(platform, qubits, args.freq_width, args.freq_step)

    # print(data.df)
    output_data = ResonatorSpectroscopyData(
        frequency=data.get_values("frequency", "Hz").to_numpy(),
        phase=data.get_values("phase", "rad").to_numpy(),
        msr=data.get_values("MSR", "V").to_numpy(),
        qubit=data.get_values("qubit").to_numpy(),
    )
    print(output_data.frequency)
    return output_data


def _resonator_spectroscopy_fit(
    data: ResonatorSpectroscopyData,
) -> ResonatorSpectroscoyResults:
    qubits = np.unique(data.qubit)

    for qubit in qubits:
        # print(qubit)
        frequency = data.frequency[data.qubit == qubit]
        print(qubit)
        print(frequency)
        # print(type(frequency))
        # print(frequency)
        # print(frequency)
    # print(np.unique(data.qubit))
    # for qubit in data.qubits:
    #     qubit_data = (
    #         data.df[data.df["qubit"] == qubit]
    #         .drop(columns=["qubit", "iteration"])
    #         .groupby("frequency", as_index=False)
    #         .mean()
    #     )
    #     frequencies_keys = parse(x)
    #     voltages_keys = parse(y)
    #     frequencies = (
    #         qubit_data[frequencies_keys[0]].pint.to(frequencies_keys[1]).pint.magnitude
    #     )  # convert frequencies to GHz for better fitting
    #     voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude

    #     # Create a lmfit model for fitting equation defined in resonator_peak
    #     model_Q = lmfit.Model(lorenzian)

    #     # Guess parameters for Lorentzian max or min
    #     if (resonator_type == "3D" and "readout_frequency" in labels[0]) or (
    #         resonator_type == "2D" and "drive_frequency" in labels[0]
    #     ):
    #         guess_center = frequencies[
    #             np.argmax(voltages)
    #         ]  # Argmax = Returns the indices of the maximum values along an axis.
    #         guess_offset = np.mean(
    #             voltages[np.abs(voltages - np.mean(voltages) < np.std(voltages))]
    #         )
    #         guess_sigma = abs(frequencies[np.argmin(voltages)] - guess_center)
    #         guess_amp = (np.max(voltages) - guess_offset) * guess_sigma * np.pi

    #     else:
    #         guess_center = frequencies[
    #             np.argmin(voltages)
    #         ]  # Argmin = Returns the indices of the minimum values along an axis.
    #         guess_offset = np.mean(
    #             voltages[np.abs(voltages - np.mean(voltages) < np.std(voltages))]
    #         )
    #         guess_sigma = abs(frequencies[np.argmax(voltages)] - guess_center)
    #         guess_amp = (np.min(voltages) - guess_offset) * guess_sigma * np.pi

    #     # Add guessed parameters to the model
    #     model_Q.set_param_hint("center", value=guess_center, vary=True)
    #     model_Q.set_param_hint("sigma", value=guess_sigma, vary=True)
    #     model_Q.set_param_hint("amplitude", value=guess_amp, vary=True)
    #     model_Q.set_param_hint("offset", value=guess_offset, vary=True)
    #     guess_parameters = model_Q.make_params()

    #     # fit the model with the data and guessed parameters
    #     try:
    #         fit_res = model_Q.fit(
    #             data=voltages, frequency=frequencies, params=guess_parameters
    #         )
    #         # get the values for postprocessing and for legend.
    #         f0 = fit_res.best_values["center"]
    #         BW = fit_res.best_values["sigma"] * 2
    #         Q = abs(f0 / BW)
    #         peak_voltage = (
    #             fit_res.best_values["amplitude"]
    #             / (fit_res.best_values["sigma"] * np.pi)
    #             + fit_res.best_values["offset"]
    #         )

    #         freq = f0 * 1e9

    #         data_fit.add(
    #             {
    #                 labels[0]: freq,
    #                 labels[1]: peak_voltage,
    #                 "popt0": fit_res.best_values["amplitude"],
    #                 "popt1": fit_res.best_values["center"],
    #                 "popt2": fit_res.best_values["sigma"],
    #                 "popt3": fit_res.best_values["offset"],
    #                 "qubit": qubit,
    #             }
    #         )
    #     except:
    #         log.warning("lorentzian_fit: the fitting was not successful")
    #         data_fit.add(
    #             {key: 0 if key != "qubit" else qubit for key in data_fit.df.columns}
    #         )


resonator_spectroscopy = Routine(
    _resonator_spectroscopy_acquisition, _resonator_spectroscopy_fit
)


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
