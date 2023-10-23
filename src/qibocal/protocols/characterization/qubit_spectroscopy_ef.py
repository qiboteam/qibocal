from dataclasses import asdict, dataclass, field

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
from .resonator_spectroscopy import ResSpecType
from .utils import GHZ_TO_HZ, HZ_TO_GHZ, spectroscopy_plot, table_dict, table_html

DEFAULT_ANHARMONICITY = 300e6
"""Initial guess for anharmonicity."""


@dataclass
class QubitSpectroscopyEFParameters(QubitSpectroscopyParameters):
    """QubitSpectroscopyEF runcard inputs."""


@dataclass
class QubitSpectroscopyEFResults(QubitSpectroscopyResults):
    """QubitSpectroscopyEF outputs."""

    anharmonicity: dict[QubitId, float] = field(default_factory=dict)


@dataclass
class QubitSpectroscopyEFData(QubitSpectroscopyData):
    """QubitSpectroscopy acquisition outputs."""

    drive_frequencies: dict[QubitId, float] = field(default_factory=dict)


def _fit_ef(data: QubitSpectroscopyEFData) -> QubitSpectroscopyEFResults:
    results = _fit(data)
    anharmoncities = {
        qubit: data.drive_frequencies[qubit] * HZ_TO_GHZ - results.frequency[qubit]
        for qubit in data.qubits
    }
    params = asdict(results)
    params.update({"anharmonicity": anharmoncities})

    return QubitSpectroscopyEFResults(**params)


def _acquisition(
    params: QubitSpectroscopyEFParameters, platform: Platform, qubits: Qubits
) -> QubitSpectroscopyEFData:
    """Data acquisition for qubit spectroscopy ef protocol.

    Similar to a qubit spectroscopy with the difference that the qubit is first
    excited to the state 1. This protocols aims at finding the transition frequency between
    state 1 and the state 2. The anharmonicity is also computed.

    If the RX12 frequency is not present in the runcard the sweep is performed around the
    qubit drive frequency shifted by DEFAULT_ANHARMONICITY, an hardcoded parameter editable
    in this file.

    """
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    rx_pulses = {}
    amplitudes = {}
    drive_frequencies = {}
    for qubit in qubits:
        rx_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        drive_frequencies[qubit] = rx_pulses[qubit].frequency
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=rx_pulses[qubit].finish, duration=params.drive_duration
        )
        if platform.qubits[qubit].native_gates.RX12.frequency is None:
            qd_pulses[qubit].frequency = (
                rx_pulses[qubit].frequency - DEFAULT_ANHARMONICITY
            )
        else:
            qd_pulses[qubit].frequency = platform.qubits[
                qubit
            ].native_gates.RX12.frequency

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
    data = QubitSpectroscopyEFData(
        resonator_type=platform.resonator_type,
        amplitudes=amplitudes,
        drive_frequencies=drive_frequencies,
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


def _plot(data: QubitSpectroscopyEFData, qubit, fit: QubitSpectroscopyEFResults):
    """Plotting function for QubitSpectroscopy."""
    figures, report = spectroscopy_plot(data, qubit, fit)
    if fit is not None:
        report = table_html(
            table_dict(
                qubit,
                ["Frequency 1->2", "Amplitude", "Anharmonicity"],
                [
                    np.round(fit.frequency[qubit] * GHZ_TO_HZ, 0),
                    fit.amplitude[qubit],
                    np.round(fit.anharmonicity[qubit] * GHZ_TO_HZ, 0),
                ],
            )
        )

    return figures, report


def _update(results: QubitSpectroscopyEFResults, platform: Platform, qubit: QubitId):
    """Update w12 frequency"""
    update.frequency_12_transition(results.frequency[qubit], platform, qubit)
    update.anharmonicity(results.anharmonicity[qubit], platform, qubit)


qubit_spectroscopy_ef = Routine(_acquisition, _fit_ef, _plot, _update)
"""QubitSpectroscopyEF Routine object."""
