from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Parameters, Results, Routine

from .resonator_spectroscopy import ResonatorSpectroscopyData, ResSpecType
from .utils import chi2_reduced, lorentzian, lorentzian_fit, spectroscopy_plot


@dataclass
class QubitSpectroscopyParameters(Parameters):
    """QubitSpectroscopy runcard inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative  to the qubit frequency."""
    freq_step: int
    """Frequency [Hz] step for sweep."""
    drive_duration: int
    """Drive pulse duration [ns]. Same for all qubits."""
    drive_amplitude: Optional[float] = None
    """Drive pulse amplitude (optional). Same for all qubits."""
    hardware_average: bool = True
    """By default hardware average will be performed."""


@dataclass
class QubitSpectroscopyResults(Results):
    """QubitSpectroscopy outputs."""

    frequency: dict[QubitId, dict[str, float]]
    """Drive frequecy [GHz] for each qubit."""
    amplitude: dict[QubitId, float]
    """Input drive amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""
    chi2_reduced: dict[QubitId, tuple[float, Optional[float]]] = field(
        default_factory=dict
    )
    """Chi2 reduced."""
    error_fit_pars: dict[QubitId, list] = field(default_factory=dict)
    """Errors of the fit parameters."""


class QubitSpectroscopyData(ResonatorSpectroscopyData):
    """QubitSpectroscopy acquisition outputs."""


def _acquisition(
    params: QubitSpectroscopyParameters, platform: Platform, targets: list[QubitId]
) -> QubitSpectroscopyData:
    """Data acquisition for qubit spectroscopy."""
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    amplitudes = {}
    for qubit in targets:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )
        if params.drive_amplitude is not None:
            qd_pulses[qubit].amplitude = params.drive_amplitude

        amplitudes[qubit] = qd_pulses[qubit].amplitude

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in targets],
        type=SweeperType.OFFSET,
    )

    # Create data structure for data acquisition.
    data = QubitSpectroscopyData(
        resonator_type=platform.resonator_type, amplitudes=amplitudes
    )

    results = platform.sweep(
        sequence,
        params.execution_parameters,
        sweeper,
    )

    # retrieve the results for every qubit
    for qubit, ro_pulse in ro_pulses.items():
        result = results[ro_pulse.serial]
        # store the results
        data.register_qubit(
            ResSpecType,
            (qubit),
            dict(
                signal=result.average.magnitude,
                phase=result.average.phase,
                freq=delta_frequency_range + qd_pulses[qubit].frequency,
                error_signal=result.average.std,
                error_phase=result.phase_std,
            ),
        )
    return data


def _fit(data: QubitSpectroscopyData) -> QubitSpectroscopyResults:
    """Post-processing function for QubitSpectroscopy."""
    qubits = data.qubits
    frequency = {}
    fitted_parameters = {}
    error_fit_pars = {}
    chi2 = {}
    for qubit in qubits:
        fit_result = lorentzian_fit(
            data[qubit], resonator_type=data.resonator_type, fit="qubit"
        )
        if fit_result is not None:
            frequency[qubit], fitted_parameters[qubit], error_fit_pars[qubit] = (
                fit_result
            )
            chi2[qubit] = (
                chi2_reduced(
                    data[qubit].signal,
                    lorentzian(data[qubit].freq, *fitted_parameters[qubit]),
                    data[qubit].error_signal,
                ),
                np.sqrt(2 / len(data[qubit].freq)),
            )
    return QubitSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        amplitude=data.amplitudes,
        error_fit_pars=error_fit_pars,
        chi2_reduced=chi2,
    )


def _plot(data: QubitSpectroscopyData, target: QubitId, fit: QubitSpectroscopyResults):
    """Plotting function for QubitSpectroscopy."""
    return spectroscopy_plot(data, target, fit)


def _update(results: QubitSpectroscopyResults, platform: Platform, target: QubitId):
    update.drive_frequency(results.frequency[target], platform, target)


qubit_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""QubitSpectroscopy Routine object."""
