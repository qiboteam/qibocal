from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Parameters, Qubits, Results, Routine

from .resonator_spectroscopy import ResonatorSpectroscopyData, ResSpecType
from .utils import PowerLevel, lorentzian_fit, spectroscopy_plot


@dataclass
class ResonatorSpectroscopyAttenuationParameters(Parameters):
    """ResonatorSpectroscopy runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    power_level: PowerLevel
    """Power regime (low or high). If low the readout frequency will be updated.
    If high both the readout frequency and the bare resonator frequency will be updated."""
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

    frequency: dict[QubitId, float]
    """Readout frequency [GHz] for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitted parameters."""
    bare_frequency: Optional[dict[QubitId, float]] = field(
        default_factory=dict,
    )
    """Bare resonator frequency [GHz] for each qubit."""
    amplitude: Optional[dict[QubitId, float]] = field(
        default_factory=dict,
    )
    """Readout amplitude for each qubit."""
    attenuation: Optional[dict[QubitId, int]] = field(
        default_factory=dict,
    )
    """Readout attenuation [dB] for each qubit."""


@dataclass
class ResonatorSpectroscopyAttenuationData(ResonatorSpectroscopyData):
    """Data structure for resonator spectroscopy with attenuation."""

    attenuations: dict[QubitId, int] = field(default_factory=dict)


def _acquisition(
    params: ResonatorSpectroscopyAttenuationParameters,
    platform: Platform,
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
        type=SweeperType.OFFSET,
    )
    data = ResonatorSpectroscopyAttenuationData(
        resonator_type=platform.resonator_type,
        power_level=params.power_level,
        amplitudes=amplitudes,
        attenuations=attenuations,
    )

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        # store the results
        data.register_qubit(
            ResSpecType,
            (qubit),
            dict(
                msr=result.magnitude,
                phase=result.phase,
                freq=delta_frequency_range + ro_pulses[qubit].frequency,
            ),
        )
    # finally, save the remaining data
    return data


def _fit(
    data: ResonatorSpectroscopyAttenuationData,
) -> ResonatorSpectroscopyAttenuationResults:
    """Post-processing function for ResonatorSpectroscopy."""
    qubits = data.qubits
    bare_frequency = {}
    frequency = {}
    fitted_parameters = {}
    for qubit in qubits:
        freq, fitted_params = lorentzian_fit(
            data[qubit], resonator_type=data.resonator_type, fit="resonator"
        )
        if data.power_level is PowerLevel.high:
            bare_frequency[qubit] = freq

        frequency[qubit] = freq
        fitted_parameters[qubit] = fitted_params
    if data.power_level is PowerLevel.high:
        return ResonatorSpectroscopyAttenuationResults(
            frequency=frequency,
            fitted_parameters=fitted_parameters,
            bare_frequency=bare_frequency,
            amplitude=data.amplitudes,
            attenuation=data.attenuations,
        )
    else:
        return ResonatorSpectroscopyAttenuationResults(
            frequency=frequency,
            fitted_parameters=fitted_parameters,
            amplitude=data.amplitudes,
            attenuation=data.attenuations,
        )


def _plot(
    data: ResonatorSpectroscopyAttenuationData,
    qubit,
    fit: ResonatorSpectroscopyAttenuationResults,
):
    """Plotting function for ResonatorSpectroscopyAttenuation."""
    return spectroscopy_plot(data, qubit, fit)


def _update(
    results: ResonatorSpectroscopyAttenuationResults, platform: Platform, qubit: QubitId
):
    update.readout_frequency(results.frequency[qubit], platform, qubit)

    # if this condition is satifisfied means that we are in the low power regime
    # therefore we update also the readout amplitude
    if len(results.bare_frequency) == 0:
        update.readout_amplitude(results.amplitude[qubit], platform, qubit)
        update.readout_attenuation(results.attenuation[qubit], platform, qubit)
    else:
        update.bare_resonator_frequency(results.bare_frequency[qubit], platform, qubit)


resonator_spectroscopy_attenuation = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorSpectroscopyAttenuation Routine object."""
