from dataclasses import asdict, dataclass, field

import numpy as np
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.update import replace

from ... import update
from ...result import magnitude, phase
from ..utils import readout_frequency, table_dict, table_html
from .qubit_spectroscopy import (
    QubitSpectroscopyData,
    QubitSpectroscopyParameters,
    QubitSpectroscopyResults,
    QubitSpecType,
    _fit,
)
from .qubit_spectroscopy import _plot as qubit_spectroscopy_plot

__all__ = ["qubit_spectroscopy_ef"]


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
    params: QubitSpectroscopyEFParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
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
    amplitudes = {}
    sweepers = []
    drive_frequencies = {}

    delta_frequency_range = np.arange(
        -params.freq_width, params.freq_width, params.freq_step
    )
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]

        qd_channel, qd_pulse = natives.RX()[0]
        qd12_channel, qd12_pulse = natives.RX12()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        qd12_pulse = replace(qd12_pulse, duration=params.drive_duration)
        if params.drive_amplitude is not None:
            qd12_pulse = replace(qd12_pulse, amplitude=params.drive_amplitude)

        amplitudes[qubit] = qd12_pulse.amplitude
        ro_pulses[qubit] = ro_pulse

        sequence.append((qd_channel, qd_pulse))
        sequence.append((qd12_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((qd12_channel, qd12_pulse))
        sequence.append(
            (ro_channel, Delay(duration=qd_pulse.duration + qd12_pulse.duration))
        )
        sequence.append((ro_channel, ro_pulse))

        drive_frequencies[qubit] = platform.calibration.single_qubits[
            qubit
        ].qubit.frequency_01
        sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                values=platform.config(qd12_channel).frequency + delta_frequency_range,
                channels=[qd12_channel],
            )
        )

    data = QubitSpectroscopyEFData(
        amplitudes=amplitudes,
        drive_frequencies=drive_frequencies,
    )

    results = platform.execute(
        [sequence],
        [sweepers],
        updates=[
            {
                platform.qubits[q].probe: {
                    "frequency": readout_frequency(q, platform, state=1)
                }
            }
            for q in targets
        ],
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.INTEGRATION,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )

    # retrieve the results for every qubit
    for qubit, ro_pulse in ro_pulses.items():
        result = results[ro_pulse.id]

        f0 = platform.config(platform.qubits[qubit].drive_extra[1, 2]).frequency

        signal = magnitude(result)
        _phase = phase(result)

        data.register_qubit(
            QubitSpecType,
            (qubit),
            dict(
                signal=signal,
                phase=_phase,
                freq=delta_frequency_range + f0,
            ),
        )
    return data


def _plot(
    data: QubitSpectroscopyEFData, target: QubitId, fit: QubitSpectroscopyEFResults
):
    """Plotting function for QubitSpectroscopy."""
    figures, report = qubit_spectroscopy_plot(data, target, fit)
    if fit is not None:
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


def _update(
    results: QubitSpectroscopyEFResults, platform: CalibrationPlatform, target: QubitId
):
    """Update w12 frequency"""
    update.frequency_12_transition(results.frequency[target], platform, target)
    platform.calibration.single_qubits[target].qubit.frequency_12 = results.frequency[
        target
    ]


qubit_spectroscopy_ef = Protocol(_acquisition, fit_ef, _plot, _update)
"""QubitSpectroscopyEF Protocol object."""
