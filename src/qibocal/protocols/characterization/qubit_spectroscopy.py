from dataclasses import dataclass
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
from .utils import lorentzian_fit, spectroscopy_plot


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


@dataclass
class QubitSpectroscopyResults(Results):
    """QubitSpectroscopy outputs."""

    frequency: dict[QubitId, dict[str, float]]
    """Drive frequecy [GHz] for each qubit."""
    amplitude: dict[QubitId, float]
    """Input drive amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


class QubitSpectroscopyData(ResonatorSpectroscopyData):
    """QubitSpectroscopy acquisition outputs."""


def _acquisition(
    params: QubitSpectroscopyParameters, platform: Platform, qubits: Qubits
) -> QubitSpectroscopyData:
    """Data acquisition for qubit spectroscopy."""
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    amplitudes = {}
    for qubit in qubits:
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
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    # Create data structure for data acquisition.
    data = QubitSpectroscopyData(
        resonator_type=platform.resonator_type, amplitudes=amplitudes
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
    for qubit, ro_pulse in ro_pulses.items():
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulse.serial]
        # store the results
        data.register_qubit(
            ResSpecType,
            (qubit),
            dict(
                msr=result.magnitude,
                phase=result.phase,
                freq=delta_frequency_range + qd_pulses[qubit].frequency,
            ),
        )
    return data


def _fit(data: QubitSpectroscopyData) -> QubitSpectroscopyResults:
    """Post-processing function for QubitSpectroscopy."""
    qubits = data.qubits
    frequency = {}
    fitted_parameters = {}
    for qubit in qubits:
        freq, fitted_params = lorentzian_fit(
            data[qubit], resonator_type=data.resonator_type, fit="qubit"
        )
        frequency[qubit] = freq
        fitted_parameters[qubit] = fitted_params

    return QubitSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        amplitude=data.amplitudes,
    )


def _plot(data: QubitSpectroscopyData, qubit, fit: QubitSpectroscopyResults):
    """Plotting function for QubitSpectroscopy."""
    return spectroscopy_plot(data, qubit, fit)


def _update(results: QubitSpectroscopyResults, platform: Platform, qubit: QubitId):
    update.drive_frequency(results.frequency[qubit], platform, qubit)


qubit_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""QubitSpectroscopy Routine object."""
