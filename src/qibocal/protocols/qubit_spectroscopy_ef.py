from dataclasses import asdict, dataclass, field

import numpy as np
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Routine

from .qubit_spectroscopy import (
    QubitSpectroscopyData,
    QubitSpectroscopyParameters,
    QubitSpectroscopyResults,
    _fit,
)
from .resonator_spectroscopy import ResSpecType
from .utils import spectroscopy_plot, table_dict, table_html

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


def fit_ef(data: QubitSpectroscopyEFData) -> QubitSpectroscopyEFResults:
    results = _fit(data)
    anharmoncities = {
        qubit: results.frequency[qubit] - data.drive_frequencies[qubit]
        for qubit in data.qubits
        if qubit in results
    }
    params = asdict(results)
    return QubitSpectroscopyEFResults(anharmonicity=anharmoncities, **params)


def _acquisition(
    params: QubitSpectroscopyEFParameters, platform: Platform, targets: list[QubitId]
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
    for qubit in targets:
        rx_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        drive_frequencies[qubit] = rx_pulses[qubit].frequency
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=rx_pulses[qubit].finish, duration=params.drive_duration
        )

        if platform.qubits[qubit].native_gates.RX12.frequency is None:

            qd_pulses[qubit].frequency = (
                rx_pulses[qubit].frequency + DEFAULT_ANHARMONICITY
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
        pulses=[qd_pulses[qubit] for qubit in targets],
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


def _plot(
    data: QubitSpectroscopyEFData, target: QubitId, fit: QubitSpectroscopyEFResults
):
    """Plotting function for QubitSpectroscopy."""
    figures, report = spectroscopy_plot(data, target, fit)
    show_error_bars = not np.isnan(data[target].error_signal).any()
    if fit is not None:
        if show_error_bars:
            report = table_html(
                table_dict(
                    target,
                    [
                        "Frequency 1->2 [Hz]",
                        "Amplitude [a.u.]",
                        "Anharmonicity [Hz]",
                        "Chi2",
                    ],
                    [
                        (fit.frequency[target], fit.error_fit_pars[target][1]),
                        (fit.amplitude[target], fit.error_fit_pars[target][0]),
                        (fit.anharmonicity[target], fit.error_fit_pars[target][2]),
                        fit.chi2_reduced[target],
                    ],
                    display_error=True,
                )
            )
        else:
            report = table_html(
                table_dict(
                    target,
                    [
                        "Frequency 1->2 [Hz]",
                        "Amplitude [a.u.]",
                        "Anharmonicity [Hz]",
                    ],
                    [
                        fit.frequency[target],
                        fit.amplitude[target],
                        fit.anharmonicity[target],
                    ],
                    display_error=False,
                )
            )

    return figures, report


def _update(results: QubitSpectroscopyEFResults, platform: Platform, target: QubitId):
    """Update w12 frequency"""
    update.frequency_12_transition(results.frequency[target], platform, target)
    update.anharmonicity(results.anharmonicity[target], platform, target)


qubit_spectroscopy_ef = Routine(_acquisition, fit_ef, _plot, _update)
"""QubitSpectroscopyEF Routine object."""
