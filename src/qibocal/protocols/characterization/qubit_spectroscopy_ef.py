from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Qubits, Routine

from .qubit_spectroscopy import (
    QubitSpectroscopyData,
    QubitSpectroscopyParameters,
    QubitSpectroscopyResults,
    _fit,
)
from .utils import spectroscopy_plot

DEFAULT_ANHARMONICITY = 300e6


@dataclass
class QubitSpectroscopyEFParameters(QubitSpectroscopyParameters):
    """QubitSpectroscopyEF runcard inputs."""


@dataclass
class QubitSpectroscopyEFResults(QubitSpectroscopyResults):
    """QubitSpectroscopyEF outputs."""


class QubitSpectroscopyEFData(QubitSpectroscopyData):
    """QubitSpectroscopy acquisition outputs."""


def _acquisition(
    params: QubitSpectroscopyEFParameters, platform: Platform, qubits: Qubits
) -> QubitSpectroscopyEFData:
    """Data acquisition for qubit spectroscopy."""
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    rx_pulses = {}
    amplitudes = {}
    for qubit in qubits:
        rx_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=rx_pulses[qubit].finish, duration=params.drive_duration
        )
        qd_pulses[qubit].frequency -= DEFAULT_ANHARMONICITY
        if params.drive_amplitude is not None:
            qd_pulses[qubit].amplitude = params.drive_amplitude

        amplitudes[qubit] = qd_pulses[qubit].amplitude

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(rx_pulses[qubit])
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # sweep only before qubit frequency
    delta_frequency_range = np.arange(
        -params.freq_width, params.freq_width, params.freq_step
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
            qubit,
            msr=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + qd_pulses[qubit].frequency,
        )
    return data


def _plot(data: QubitSpectroscopyEFData, qubit, fit: QubitSpectroscopyEFResults):
    """Plotting function for QubitSpectroscopy."""
    figures, report = spectroscopy_plot(data, qubit, fit)
    if report is not None:
        report = report.replace("qubit frequency", "w12 frequency")

    return figures, report


def _update(results: QubitSpectroscopyEFResults, platform: Platform, qubit: QubitId):
    """Update w12 frequency"""
    update.frequency_12_transition(results.frequency[qubit], platform, qubit)


qubit_spectroscopy_ef = Routine(_acquisition, _fit, _plot, _update)
"""QubitSpectroscopy Routine object."""
