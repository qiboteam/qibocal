from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from .utils import PowerLevel, lorentzian_fit, spectroscopy_plot


@dataclass
class ResonatorSpectroscopyParameters(Parameters):
    freq_width: int
    freq_step: int
    nshots: int
    power_level: PowerLevel
    relaxation_time: int
    amplitude: Optional[float] = None
    attenuation: Optional[int] = None

    def __post_init__(self):
        if self.attenuation is not None and self.amplitude is not None:
            raise ValueError(
                "Cannot specify attenuation and amplitude at the same time."
            )
        # TODO: ask Alessandro if there is a proper way to pass Enum to class
        self.power_level = PowerLevel(self.power_level)


@dataclass
class ResonatorSpectroscopyResults(Results):
    frequency: Dict[List[Tuple], str] = field(metadata=dict(update="readout_frequency"))
    fitted_parameters: Dict[List[Tuple], List]
    bare_frequency: Optional[Dict[List[Tuple], str]] = field(
        default_factory=dict, metadata=dict(update="bare_resonator_frequency")
    )
    amplitude: Optional[Dict[List[Tuple], str]] = field(
        default_factory=dict, metadata=dict(update="readout_amplitude")
    )
    attenuation: Optional[Dict[List[Tuple], str]] = field(
        default_factory=dict, metadata=dict(update="readout_attenuation")
    )


class ResonatorSpectroscopyData(DataUnits):
    def __init__(
        self, resonator_type, power_level=None, amplitude=None, attenuation=None
    ):
        super().__init__(
            "data",
            {"frequency": "Hz"},
            options=["qubit"],
        )
        self._resonator_type = resonator_type
        self._power_level = power_level
        self._amplitude = amplitude
        self._attenuation = attenuation

    @property
    def resonator_type(self):
        """Type of resonator"""
        return self._resonator_type

    @property
    def power_level(self):
        """Resonator spectroscopy power level"""
        return self._power_level

    @property
    def amplitude(self):
        """Readout pulse amplitude common for all qubits"""
        return self._amplitude

    @property
    def attenuation(self):
        """Attenuation value common for all qubits"""
        return self._attenuation


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: ResonatorSpectroscopyParameters
) -> ResonatorSpectroscopyData:
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        if params.amplitude is not None:
            ro_pulses[qubit].amplitude = params.amplitude
        else:
            params.amplitude = ro_pulses[qubit].amplitude
        if params.attenuation is not None:
            platform.set_attenuation(qubit, params.attenuation)

        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
    )
    data = ResonatorSpectroscopyData(
        platform.resonator_type,
        params.power_level,
        params.amplitude,
        params.attenuation,
    )
    results = platform.sweep(
        sequence,
        sweeper,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        # store the results
        r = result.raw
        r.update(
            {
                "frequency[Hz]": delta_frequency_range + ro_pulses[qubit].frequency,
                "qubit": len(delta_frequency_range) * [qubit],
            }
        )
        data.add_data_from_dict(r)
    # finally, save the remaining data
    return data


def _fit(data: ResonatorSpectroscopyData) -> ResonatorSpectroscopyResults:
    return ResonatorSpectroscopyResults(**lorentzian_fit(data))


def _plot(data: ResonatorSpectroscopyData, fit: ResonatorSpectroscopyResults, qubit):
    return spectroscopy_plot(data, fit, qubit)


resonator_spectroscopy = Routine(_acquisition, _fit, _plot)
