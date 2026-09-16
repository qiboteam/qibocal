"""Ramsey experiment on the 1 -> 2 transition."""

from dataclasses import dataclass

import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Platform,
    PulseId,
    PulseLike,
    PulseSequence,
    Readout,
    Sweeper,
)

from qibocal import update
from qibocal.auto.operation import Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.result import unpack
from qibocal.update import replace

from ..utils import readout_frequency
from .acquisition import RamseyParameters, RamseyResults
from .processing import signal_plot
from .signal import RamseySignalData, RamseySignalType, _fit

__all__ = ["ramsey_ef_signal"]


@dataclass
class RamseyEFParameters(RamseyParameters):
    """RamseyEF runcard inputs."""


@dataclass
class RamseyEFResults(RamseyResults):
    """RamseyEF outputs."""


@dataclass
class RamseyEFData(RamseySignalData):
    """RamseyEF acquisition outputs."""


def ramsey_ef_sequence(
    platform: Platform,
    targets: list[QubitId],
    wait: int = 0,
) -> tuple[PulseSequence, dict[QubitId, Readout], list[PulseLike]]:
    """Pulse sequence used in the Ramsey experiment on the 1 -> 2 transition.

    The qubit is first excited to state 1, then the Ramsey sequence is played
    on the 1 -> 2 transition:

    RX -- RX12_90 -- wait -- RX12_90 -- MZ

    The RX12_90 pulse is obtained by halving the amplitude of the calibrated
    RX12 pulse.
    This is an approximation that might be more or less accurate depending on the pulse shape.
    Currently only rectangular pulses are supported so the approximation should be good enough.

    Returns the sequence, the readout pulses (needed to retrieve the results)
    and the delays to be swept.
    """
    sequence = PulseSequence()
    ro_pulses: dict[QubitId, Readout] = {}
    delays: list[PulseLike] = []

    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        assert natives.RX12 is not None, f"Missing RX12 calibration for qubit {qubit}."

        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]
        qd12_channel = platform.qubits[qubit].drive_extra[1, 2]
        [(_, rx12_pulse)] = natives.RX12()

        # pi/2 rotation on the 1 -> 2 transition
        rx12_90_pulse = replace(rx12_pulse, amplitude=rx12_pulse.amplitude / 2)

        # one delay per channel involved, all swept together
        waits = [Delay(duration=wait) for _ in range(2)]
        delays.extend(waits)
        ro_pulses[qubit] = ro_pulse

        sequence.append((qd_channel, qd_pulse))
        sequence.extend(
            [
                (qd12_channel, Delay(duration=qd_pulse.duration)),
                (qd12_channel, rx12_90_pulse),
                (qd12_channel, waits[0]),
                # new id, otherwise the two pulses would not be distinguishable
                (qd12_channel, rx12_90_pulse.new()),
            ]
        )
        sequence.extend(
            [
                (
                    ro_channel,
                    Delay(duration=qd_pulse.duration + 2 * rx12_90_pulse.duration),
                ),
                (ro_channel, waits[1]),
                (ro_channel, ro_pulse),
            ]
        )

    return sequence, ro_pulses, delays


def execute_ef_experiment(
    sequence: PulseSequence,
    delays: list[PulseLike],
    platform: CalibrationPlatform,
    targets: list[QubitId],
    params: RamseyEFParameters,
) -> tuple[dict[PulseId, Results], npt.NDArray]:
    """Execute the Ramsey EF experiment on the platform.

    The readout is performed at the frequency corresponding to state 1 and,
    when requested, the 1 -> 2 drive channel is detuned.
    """
    updates = [
        {
            platform.qubits[qubit].probe: {
                "frequency": readout_frequency(qubit, platform, state=1)
            }
        }
        for qubit in targets
    ]

    if params.detuning is not None:
        for qubit in targets:
            channel = platform.qubits[qubit].drive_extra[1, 2]
            f0 = platform.config(channel).frequency
            updates.append({channel: {"frequency": f0 + params.detuning}})

    sweeper = Sweeper(
        parameter=Parameter.duration,
        range=params.delay_range,
        pulses=delays,
    )

    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        updates=updates,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    return results, sweeper.values


def _acquisition(
    params: RamseyEFParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RamseyEFData:
    """Data acquisition for Ramsey EF experiment (detuned)."""

    data = RamseyEFData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.config(platform.qubits[qubit].drive_extra[1, 2]).frequency
            for qubit in targets
        },
    )

    sequence, ro_pulses, delays = ramsey_ef_sequence(platform, targets)

    results, waits = execute_ef_experiment(
        sequence=sequence,
        delays=delays,
        platform=platform,
        targets=targets,
        params=params,
    )

    for qubit in targets:
        i, q = unpack(results[ro_pulses[qubit].id])
        data.register_qubit(
            RamseySignalType,
            (qubit),
            {
                "wait": waits,
                "i": i,
                "q": q,
            },
        )

    return data


def _plot(
    data: RamseyEFData, target: QubitId, fit: RamseyEFResults | None = None
) -> tuple[list[go.Figure], str]:
    """Plotting function for Ramsey EF experiment."""

    figures, report = signal_plot(
        waits=data.waits,
        signal=data.qubit_signal(target),
        target=target,
        fit=fit,
        yaxis_title="Signal [a.u.]",
    )
    if report is not None:
        report = report.replace("Drive Frequency", "Drive Frequency 1->2")
    return figures, report


def _update(results: RamseyEFResults, platform: CalibrationPlatform, target: QubitId):
    """Update the 1 -> 2 transition frequency."""
    if results.detuning is None:
        # without detuning only T2 is measured, and there is no place
        # to store the T2 of the 1 -> 2 transition
        return
    update.frequency_12_transition(results.frequency[target][0], platform, target)


ramsey_ef_signal = Protocol(_acquisition, _fit, _plot, _update)
"""Ramsey EF Protocol object.

The protocol consists in applying the following pulse sequence:
RX - RX12_90 - wait - RX12_90 - MZ
for different waiting times `wait`, measuring at the readout frequency
associated to state 1.
The range of waiting times is defined through the attributes
`delay_between_pulses_*` available in `RamseyParameters`. The final range
will be constructed using `np.arange`.
It is possible to detune the 1 -> 2 drive frequency using the parameter
`detuning` in `RamseyParameters`, which will increment the frequency of the
1 -> 2 drive channel accordingly.
The following protocol will display on the y-axis the signal amplitude.
"""
