from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qibolab import Delay, Parameter, PulseSequence, Sweeper

from qibocal.auto.operation import Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude, phase
from qibocal.update import replace

from ... import update
from ..resonator_spectroscopies.resonator_spectroscopy import (
    ResonatorSpectroscopyData,
    ResSpecType,
)
from ..resonator_spectroscopies.resonator_utils import spectroscopy_plot
from ..utils import (
    chi2_reduced,
    lorentzian,
    lorentzian_fit,
    readout_frequency,
)

__all__ = [
    "qubit_spectroscopy",
    "QubitSpectroscopyParameters",
    "QubitSpectroscopyResults",
    "QubitSpectroscopyData",
    "_fit",
]


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
    params: QubitSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitSpectroscopyData:
    """Data acquisition for qubit spectroscopy."""
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    amplitudes = {}
    sweepers = []
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        qd_pulse = replace(qd_pulse, duration=params.drive_duration)
        if params.drive_amplitude is not None:
            qd_pulse = replace(qd_pulse, amplitude=params.drive_amplitude)

        amplitudes[qubit] = qd_pulse.amplitude
        qd_pulses[qubit] = qd_pulse
        ro_pulses[qubit] = ro_pulse

        sequence.append((qd_channel, qd_pulse))
        sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((ro_channel, ro_pulse))

        f0 = platform.config(qd_channel).frequency
        sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                values=f0 + delta_frequency_range,
                channels=[qd_channel],
            )
        )

    # Create data structure for data acquisition.
    data = QubitSpectroscopyData(
        resonator_type=platform.resonator_type, amplitudes=amplitudes
    )

    results = platform.execute(
        [sequence],
        [sweepers],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in targets
        ],
        **params.execution_parameters,
    )

    # retrieve the results for every qubit
    for qubit, ro_pulse in ro_pulses.items():
        result = results[ro_pulse.id]
        # store the results
        f0 = platform.config(platform.qubits[qubit].drive).frequency
        signal = magnitude(result)
        _phase = phase(result)
        if len(signal.shape) > 1:
            error_signal = np.std(signal, axis=0, ddof=1) / np.sqrt(signal.shape[0])
            signal = np.mean(signal, axis=0)
            error_phase = np.std(_phase, axis=0, ddof=1) / np.sqrt(_phase.shape[0])
            _phase = np.mean(_phase, axis=0)
        else:
            error_signal, error_phase = None, None
        data.register_qubit(
            ResSpecType,
            (qubit),
            dict(
                signal=signal,
                phase=_phase,
                freq=delta_frequency_range + f0,
                error_signal=error_signal,
                error_phase=error_phase,
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


def _update(
    results: QubitSpectroscopyResults, platform: CalibrationPlatform, target: QubitId
):
    platform.calibration.single_qubits[target].qubit.frequency_01 = results.frequency[
        target
    ]
    update.drive_frequency(results.frequency[target], platform, target)


qubit_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""QubitSpectroscopy Routine object."""
