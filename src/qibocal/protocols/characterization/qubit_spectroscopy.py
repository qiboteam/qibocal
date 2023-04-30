from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from .resonator_spectroscopy import ResonatorSpectroscopyData
from .utils import lorentzian_fit, spectroscopy_plot


@dataclass
class QubitSpectroscopyParameters(Parameters):
    freq_width: int
    freq_step: int
    drive_duration: int
    drive_amplitude: Optional[float] = None
    nshots: int = 1024
    relaxation_time: int = 50


@dataclass
class QubitSpectroscopyResults(Results):
    frequency: Dict[List[Tuple], str] = field(metadata=dict(update="drive_frequency"))
    amplitude: Dict[List[Tuple], str]
    fitted_parameters: Dict[List[Tuple], List]
    attenuation: Optional[Dict[List[Tuple], str]] = field(
        default_factory=dict,
    )


class QubitSpectroscopyData(ResonatorSpectroscopyData):
    """Data Structure for qubit spectroscopy"""


def _acquisition(
    params: QubitSpectroscopyParameters, platform: AbstractPlatform, qubits: Qubits
) -> QubitSpectroscopyData:
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )
        if params.drive_amplitude is not None:
            qd_pulses[qubit].amplitude = params.drive_amplitude
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit frequency
    data = QubitSpectroscopyData(
        platform.resonator_type, amplitude=params.drive_amplitude
    )

    results = platform.sweep(
        sequence,
        sweeper,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )

    # retrieve the results for every qubit
    for qubit, ro_pulse in ro_pulses.items():
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulse.serial]
        r = result.raw
        # store the results
        r.update(
            {
                "frequency[Hz]": delta_frequency_range + qd_pulses[qubit].frequency,
                "qubit": len(delta_frequency_range) * [qubit],
            }
        )
        data.add_data_from_dict(r)

        # finally, save the remaining data and fits
    return data


def _fit(data: QubitSpectroscopyData) -> QubitSpectroscopyResults:
    qubits = data.df["qubit"].unique()
    bare_frequency = {}
    amplitudes = {}
    attenuations = {}
    frequency = {}
    fitted_parameters = {}
    for qubit in qubits:
        freq, fitted_params = lorentzian_fit(data, qubit)
        frequency[qubit] = freq
        amplitudes[qubit] = data.amplitude
        attenuations[qubit] = data.attenuation
        fitted_parameters[qubit] = fitted_params

    return QubitSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        amplitude=amplitudes,
        attenuation=attenuations,
    )


def _plot(data: QubitSpectroscopyData, fit: QubitSpectroscopyResults, qubit):
    return spectroscopy_plot(data, fit, qubit)


qubit_spectroscopy = Routine(_acquisition, _fit, _plot)
