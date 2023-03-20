from dataclasses import dataclass, field
from enum import Enum

import lmfit
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits


@dataclass
class ResonatorSpectroscopyParameters(Parameters):
    freq_width: int
    freq_step: int
    software_averages: int


@dataclass
class ResonatorSpectrscopyResults(Results):
    frequency: str = field(metadata=dict(update="readout_frequency"))


class ResonatorSpectroscopyData(DataUnits):
    def __init__(self):
        super().__init__(
            "data",
            {"frequency": "Hz"},
            options=["qubit", "iteration", "resonator_type"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: ResonatorSpectroscopyParameters
) -> ResonatorSpectroscopyData:
    # reload instrument settings from runcard
    platform.reload_settings()
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:

    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    # save runcard local oscillator frequencies to be able to calculate new intermediate frequencies
    # lo_frequencies = {qubit: platform.get_lo_frequency(qubit) for qubit in qubits}

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency
    data = ResonatorSpectroscopyData()

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(params.software_averages):
        # sweep the parameter
        for delta_freq in delta_frequency_range:
            # change freq of readout pulse
            for qubit in qubits:
                ro_pulses[qubit].frequency = (
                    delta_freq + qubits[qubit].readout_frequency
                )

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            # retrieve the results for every qubit
            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].to_dict()
                # store the results
                r.update(
                    {
                        "frequency[Hz]": ro_pulse.frequency,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                        "resonator_type": platform.resonator_type,
                    }
                )
                data.add(r)
            count += 1
    # finally, save the remaining data and fits
    return data


def lorentzian(frequency, amplitude, center, sigma, offset):
    # http://openafox.com/science/peak-function-derivations.html
    return (amplitude / np.pi) * (
        sigma / ((frequency - center) ** 2 + sigma**2)
    ) + offset


def _fit(data: ResonatorSpectroscopyData) -> Results:
    qubits = data.df["qubit"].unique()
    resonator_type = data.df["resonator_type"].unique()

    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration", "resonator_type"])
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

        return ResonatorSpectrscopyResults(frequency=f0)


resonator_spectroscopy = Routine(_acquisition, _fit)


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
