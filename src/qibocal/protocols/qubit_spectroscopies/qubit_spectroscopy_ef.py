from dataclasses import dataclass, field

import numpy as np
from qibolab import (
    AcquisitionType,
    AveragingMode,
    IqConfig,
    ParallelSweepers,
    Parameter,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform

from ... import update
from ..utils import Range, readout_frequency, table_dict, table_html
from .qubit_spectroscopy import (
    QubitSpectroscopyData,
    QubitSpectroscopyParameters,
    QubitSpectroscopyResults,
    _lorentzian_fit,
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

    drive: dict[QubitId, float]
    """Frequency used for the drive pi pulse."""
    drive12_ranges: dict[QubitId, Range]
    """Frequency ranges for the spectroscopic tone."""


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
    ro_pulses: dict[QubitId, Readout] = {}
    # TODO: remove, and propagate differently, since it is always the same for
    # all qubits
    amplitudes: dict[QubitId, float] = {q: params.drive_amplitude for q in targets}
    sweepers: ParallelSweepers = []
    drive_frequencies: dict[QubitId, list[float]] = {}
    drive: dict[QubitId, float] = {}
    drive12_ranges: dict[QubitId, Range] = {}

    freq_range = params.frequency_range()
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]

        assert natives.RX is not None
        assert natives.MZ is not None
        qd_channel = platform.qubits[qubit].drive
        qd12_channel = platform.qubits[qubit].drive_extra[1, 2]
        assert qd_channel is not None
        readout = natives.MZ()

        ro = readout[0][1]
        assert isinstance(ro, Readout)
        ro_pulses[qubit] = ro
        qd12_pulse = Pulse(
            duration=params.drive_duration,
            amplitude=params.drive_amplitude,
            envelope=Rectangular(),
        )

        seq = PulseSequence()
        seq |= natives.RX()
        seq |= [(qd12_channel, qd12_pulse)]
        seq |= readout
        sequence.extend(seq)

        qd_config = platform.config(qd_channel)
        qd12_config = platform.config(qd12_channel)
        assert isinstance(qd_config, IqConfig)
        assert isinstance(qd12_config, IqConfig)
        drive[qubit] = qd_config.frequency
        f12 = qd12_config.frequency
        f12_range = (f12 + freq_range[0], f12 + freq_range[1], freq_range[2])
        drive12_ranges[qubit] = f12_range
        # TODO: it makes no sense to propagate the unraveled range, but it is to
        # match the format of the plotting function from the qubit spectroscopy
        drive_frequencies[qubit] = np.arange(*f12_range).tolist()
        sweepers.append(
            Sweeper(
                parameter=Parameter.frequency, range=f12_range, channels=[qd12_channel]
            )
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
    return QubitSpectroscopyEFData(
        amplitudes=amplitudes,
        drive_frequencies=drive_frequencies,
        drive12_ranges=drive12_ranges,
        drive=drive,
        data={q: results[ro_pulses[q].id] for q in targets},
    )


def _fit(data: QubitSpectroscopyEFData) -> QubitSpectroscopyEFResults:
    frequency: dict[QubitId, float] = {}
    params: dict[QubitId, list[float]] = {}
    for qubit in data.qubits:
        freqs = np.arange(*data.drive12_ranges[qubit])
        frequency[qubit], params[qubit], _ = _lorentzian_fit(freqs, data.signal(qubit))

    anharmonicities = {
        qubit: frequency[qubit] - data.drive[qubit] for qubit in data.qubits
    }
    return QubitSpectroscopyEFResults(
        frequency=frequency,
        amplitude=data.amplitudes,
        fitted_parameters=params,
        anharmonicity=anharmonicities,
    )


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


qubit_spectroscopy_ef = Protocol(_acquisition, _fit, _plot, _update)
"""QubitSpectroscopyEF Protocol object."""
