from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from .utils import PowerLevel, lorentzian_fit, spectroscopy_plot


@dataclass
class ResonatorSpectroscopyAttenuationParameters(Parameters):
    """ResonatorSpectroscopy runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    nshots: int
    """Number of shots."""
    power_level: PowerLevel
    """Power regime (low or high). If low the readout frequency will be updated.
    If high both the readout frequency and the bare resonator frequency will be updated."""
    relaxation_time: int
    """Relaxation time (ns)."""
    amplitude: Optional[float] = None
    """Readout amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    attenuation: Optional[int] = None
    """Readout attenuation (optional). If defined, same attenuation will be used in all qubits.
    Otherwise the default attenuation defined on the platform runcard will be used"""

    def __post_init__(self):
        # TODO: ask Alessandro if there is a proper way to pass Enum to class
        self.power_level = PowerLevel(self.power_level)


@dataclass
class ResonatorSpectroscopyAttenuationResults(Results):
    """ResonatorSpectroscopy outputs."""

    frequency: Dict[Union[str, int], float] = field(
        metadata=dict(update="readout_frequency")
    )
    """Readout frequency [GHz] for each qubit."""
    fitted_parameters: Dict[Union[str, int], Dict[str, float]]
    """Raw fitted parameters."""
    bare_frequency: Optional[Dict[Union[str, int], float]] = field(
        default_factory=dict, metadata=dict(update="bare_resonator_frequency")
    )
    """Bare resonator frequency [GHz] for each qubit."""
    amplitude: Optional[Dict[Union[str, int], float]] = field(
        default_factory=dict, metadata=dict(update="readout_amplitude")
    )
    """Readout amplitude for each qubit."""
    attenuation: Optional[Dict[Union[str, int], int]] = field(
        default_factory=dict, metadata=dict(update="readout_attenuation")
    )
    """Readout attenuation [dB] for each qubit."""


class ResonatorSpectroscopyAttenuationData(DataUnits):
    """ResonatorSpectroscopy acquisition outputs."""

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
    params: ResonatorSpectroscopyAttenuationParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> ResonatorSpectroscopyAttenuationData:
    """Data acquisition for resonator spectroscopy attenuation."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    amplitudes = {}
    attenuations = {}

    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        if params.amplitude is not None:
            ro_pulses[qubit].amplitude = params.amplitude

        amplitudes[qubit] = ro_pulses[qubit].amplitude

        if params.attenuation is not None:
            platform.set_attenuation(qubit, params.attenuation)

        attenuations[qubit] = platform.get_attenuation(qubit)

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
    data = ResonatorSpectroscopyAttenuationData(
        platform.resonator_type,
        params.power_level,
        amplitudes,
        attenuations,
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


def _fit(
    data: ResonatorSpectroscopyAttenuationData,
) -> ResonatorSpectroscopyAttenuationResults:
    """Post-processing function for ResonatorSpectroscopy."""
    qubits = data.df["qubit"].unique()
    bare_frequency = {}
    amplitudes = {}
    attenuations = {}
    frequency = {}
    fitted_parameters = {}
    for qubit in qubits:
        freq, fitted_params = lorentzian_fit(data, qubit)
        if data.power_level is PowerLevel.high:
            bare_frequency[qubit] = freq

        frequency[qubit] = freq
        amplitudes[qubit] = data.amplitude[qubit]
        attenuations[qubit] = data.attenuation[qubit]
        fitted_parameters[qubit] = fitted_params
    if data.power_level is PowerLevel.high:
        return ResonatorSpectroscopyAttenuationResults(
            frequency=frequency,
            fitted_parameters=fitted_parameters,
            bare_frequency=bare_frequency,
            amplitude=amplitudes,
            attenuation=attenuations,
        )
    else:
        return ResonatorSpectroscopyAttenuationResults(
            frequency=frequency,
            fitted_parameters=fitted_parameters,
            amplitude=amplitudes,
            attenuation=attenuations,
        )


def _plot(
    data: ResonatorSpectroscopyAttenuationData,
    fit: ResonatorSpectroscopyAttenuationResults,
    qubit,
):
    """Plotting function for ResonatorSpectroscopyAttenuation."""
    return spectroscopy_plot(data, fit, qubit)


resonator_spectroscopy_attenuation = Routine(_acquisition, _fit, _plot)
"""ResonatorSpectroscopyAttenuation Routine object."""
