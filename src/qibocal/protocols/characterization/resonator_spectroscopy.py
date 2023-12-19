from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from .utils import (
    PowerLevel,
    chi2_reduced,
    lorentzian,
    lorentzian_fit,
    spectroscopy_plot,
)

ResSpecType = np.dtype(
    [
        ("freq", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
        ("error_signal", np.float64),
        ("error_phase", np.float64),
    ]
)
"""Custom dtype for resonator spectroscopy."""


@dataclass
class ResonatorSpectroscopyParameters(Parameters):
    """ResonatorSpectroscopy runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    power_level: Union[PowerLevel, str]
    """Power regime (low or high). If low the readout frequency will be updated.
    If high both the readout frequency and the bare resonator frequency will be updated."""
    amplitude: Optional[float] = None
    """Readout amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    attenuation: Optional[int] = None
    """Readout attenuation (optional). If defined, same attenuation will be used in all qubits.
    Otherwise the default attenuation defined on the platform runcard will be used"""

    def __post_init__(self):
        if isinstance(self.power_level, str):
            self.power_level = PowerLevel(self.power_level)


@dataclass
class ResonatorSpectroscopyResults(Results):
    """ResonatorSpectroscopy outputs."""

    frequency: dict[QubitId, float]
    """Readout frequency [GHz] for each qubit."""
    fitted_parameters: dict[QubitId, list[float]]
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
    error_fit_pars: dict[QubitId, list] = field(default_factory=dict)
    """Errors of the fit parameters."""
    chi2_reduced: dict[QubitId, tuple[float, Optional[float]]] = field(
        default_factory=dict
    )
    """Chi2 reduced."""


@dataclass
class ResonatorSpectroscopyData(Data):
    """Data structure for resonator spectroscopy with attenuation."""

    resonator_type: str
    """Resonator type."""
    amplitudes: dict[QubitId, float]
    """Amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray[ResSpecType]] = field(default_factory=dict)
    """Raw data acquired."""
    power_level: Optional[PowerLevel] = None
    """Power regime of the resonator."""
    attenuations: dict[QubitId, int] = field(default_factory=dict)
    """Readout attenuation [dB] for each qubit"""

    @classmethod
    def load(cls, path):
        obj = super().load(path)
        # Instantiate PowerLevel object
        if obj.power_level is not None:  # pylint: disable=E1101
            obj.power_level = PowerLevel(obj.power_level)  # pylint: disable=E1101
        return obj


def _acquisition(
    params: ResonatorSpectroscopyParameters,
    platform: Platform,
    qubits: Qubits,
) -> ResonatorSpectroscopyData:
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

        try:
            attenuation = platform.get_attenuation(qubit)
        except AttributeError:
            attenuation = None

        attenuations[qubit] = attenuation
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
    data = ResonatorSpectroscopyData(
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
            averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweeper,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]
        # store the results
        data.register_qubit(
            ResSpecType,
            (qubit),
            dict(
                signal=np.abs(result.average.voltage),
                phase=np.mean(result.phase, axis=0),
                freq=delta_frequency_range + ro_pulses[qubit].frequency,
                error_signal=result.average.std,
                error_phase=np.std(result.phase, ddof=1),
            ),
        )
    # finally, save the remaining data
    return data


def _fit(
    data: ResonatorSpectroscopyData,
) -> ResonatorSpectroscopyResults:
    """Post-processing function for ResonatorSpectroscopy."""
    qubits = data.qubits
    bare_frequency = {}
    frequency = {}
    fitted_parameters = {}
    error_fit_pars = {}
    chi2 = {}
    amplitudes = {}
    for qubit in qubits:
        freq, fitted_params, perr = lorentzian_fit(
            data[qubit], resonator_type=data.resonator_type, fit="resonator"
        )
        if data.power_level is PowerLevel.high:
            bare_frequency[qubit] = freq

        frequency[qubit] = freq
        fitted_parameters[qubit] = fitted_params
        error_fit_pars[qubit] = perr
        chi2[qubit] = (
            chi2_reduced(
                data[qubit].freq,
                lorentzian(data[qubit].freq, *fitted_params),
                data[qubit].error_signal,
            ),
            np.sqrt(2 / len(data[qubit].freq)),
        )
        amplitudes[qubit] = fitted_parameters[qubit][0]

    if data.power_level is PowerLevel.high:
        return ResonatorSpectroscopyResults(
            frequency=frequency,
            fitted_parameters=fitted_parameters,
            bare_frequency=bare_frequency,
            amplitude=amplitudes,
            error_fit_pars=error_fit_pars,
            chi2_reduced=chi2,
            attenuation=data.attenuations,
        )
    else:
        return ResonatorSpectroscopyResults(
            frequency=frequency,
            fitted_parameters=fitted_parameters,
            amplitude=amplitudes,
            error_fit_pars=error_fit_pars,
            chi2_reduced=chi2,
            attenuation=data.attenuations,
        )


def _plot(
    data: ResonatorSpectroscopyData,
    qubit,
    fit: ResonatorSpectroscopyResults,
):
    """Plotting function for ResonatorSpectroscopyAttenuation."""
    return spectroscopy_plot(data, qubit, fit)


def _update(results: ResonatorSpectroscopyResults, platform: Platform, qubit: QubitId):
    update.readout_frequency(results.frequency[qubit], platform, qubit)

    # if this condition is satifisfied means that we are in the low power regime
    # therefore we update also the readout amplitude
    if len(results.bare_frequency) == 0:
        update.readout_amplitude(results.amplitude[qubit], platform, qubit)
        if results.attenuation[qubit] is not None:
            update.readout_attenuation(results.attenuation[qubit], platform, qubit)
    else:
        update.bare_resonator_frequency(results.bare_frequency[qubit], platform, qubit)


resonator_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorSpectroscopyAttenuation Routine object."""
