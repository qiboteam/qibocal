"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    ParallelSweepers,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from ..utils import order_pair
from .utils import COLORAXIS

__all__ = ["chevron_couplers"]


@dataclass
class ChevronCouplersFreqParameters(Parameters):
    """ChevronCouplers runcard inputs."""

    duration_range: tuple[float, float, float]
    """Flux pulse duration range."""
    detuning_range: tuple[float, float, float]
    """Flux pulse frequency detuning range."""
    coupler_amplitude: float
    """Amplitude of the coupler flux pulse."""
    native: Literal["CZ", "iSWAP"]
    """Two qubit interaction to be calibrated."""
    dt: int | None = 0
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""


@dataclass
class ChevronCouplersFreqResults(Results):
    """ChevronCouplers results"""


ChevronFreqType = np.dtype(
    [
        ("freq", np.float64),
        ("length", np.float64),
        ("prob_high", np.float64),
        ("prob_low", np.float64),
    ]
)
"""Custom dtype for Chevron for coupler when sweeping over pulse frequency."""


@dataclass
class ChevronCouplersFreqData(Data):
    native: Literal["CZ", "iSWAP"]
    data: dict[QubitPairId, NDArray[ChevronFreqType]] = field(default_factory=dict)


def _aquisition(
    params: ChevronCouplersFreqParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> ChevronCouplersFreqData:
    r"""Perform an CZ experiment between pairs of qubits by changing its
    frequency.

    Args:
        platform: CalibrationPlatform to use.
        params: Experiment parameters.
        targets (list): List of pairs to use sequentially.

    Returns:
        ChevronData: Acquisition data.
    """

    # create a DataUnits object to store the results
    data = ChevronCouplersFreqData(native=params.native)
    total_sequence = PulseSequence()
    flux_sweepers: ParallelSweepers = []
    duration_sweepers: ParallelSweepers = []
    for pair in targets:
        cal_pair = pair if pair in platform.calibration.two_qubits else pair[::-1]
        coupler_name = platform.calibration.two_qubits[cal_pair]

        assert coupler_name is not None, (
            f"No coupler is associated to {pair} qubits pair."
        )

        coupler_channel = platform.couplers[coupler_name].flux

        # order the qubits so that the low frequency one is the first
        ordered_pair = order_pair(pair, platform)
        low_q, high_q = ordered_pair

        high_drive_freq = platform.config(platform.qubits[high_q].drive)
        low_drive_freq = platform.config(platform.qubits[low_q].drive)

        coupler_flux_freq = high_drive_freq - low_drive_freq

        high_natives = platform.natives.single_qubit[high_q]
        ro_high_channel, _ = high_natives.MZ()[0]
        drive_high_channel, drive_high_pulse = high_natives.RX()[0]

        low_natives = platform.natives.single_qubit[low_q]
        ro_low_channel, _ = low_natives.MZ()[0]
        drive_low_channel, drive_low_pulse = low_natives.RX()[0]

        single_q_seq = PulseSequence()

        if params.native == "CZ":
            low_anharm = platform.calibration.single_qubits[low_q].qubit.anharmonicity
            coupler_flux_freq -= low_anharm

            single_q_seq += [(drive_low_channel, drive_low_pulse)]

        single_q_seq += [(drive_high_channel, drive_high_pulse)]

        coupler_flux_pulse = Pulse(
            duration=params.duration_range[0],
            amplitude=params.coupler_amplitude,
            envelope=Rectangular(),
        )

        single_q_delays = [Delay(duration=params.duration_range) for _ in range(4)]
        single_q_seq += [
            (coupler_channel, coupler_flux_pulse),
            (drive_low_channel, single_q_delays[0]),
            (drive_high_channel, single_q_delays[1]),
            (ro_low_channel, single_q_delays[2]),
            (ro_high_channel, single_q_delays[3]),
        ]

        if params.native == "CZ":
            single_q_seq |= [
                (drive_low_channel, drive_low_pulse),
                (drive_high_channel, drive_high_pulse),
            ]

        single_q_seq |= high_natives.MZ() + low_natives.MZ()

        flux_sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                range=params.detuning_range,
                channels=[coupler_channel],
            )
            + coupler_flux_freq
        )

        total_sequence += single_q_seq

        duration_sweepers.append(
            Sweeper(
                parameter=Parameter.duration,
                range=params.duration_range,
                pulses=[coupler_flux_pulse] + single_q_delays,
            )
            + params.dt
        )

    results = platform.execute(
        [total_sequence],
        [duration_sweepers, flux_sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    detuning_values = np.arange(*params.detuning_range)
    duration_values = np.arange(*params.duration_range)
    for pair in targets:
        low_q, high_q = order_pair(targets)

        ro_high = list(total_sequence.channel(platform.qubits[high_q].acquisition))[-1]
        ro_low = list(total_sequence.channel(platform.qubits[low_q].acquisition))[-1]

        data.register_qubit(
            low_q,
            high_q,
            duration_values,
            detuning_values,
            results[ro_low.id],
            results[ro_high.id],
        )

    return data


def _fit(data: ChevronCouplersFreqData) -> ChevronCouplersFreqResults:
    return ChevronCouplersFreqResults()


def _plot(
    data: ChevronCouplersFreqData,
    fit: ChevronCouplersFreqParameters,
    target: QubitPairId,
):
    """Plot the experiment result for a single pair."""
    # reverse qubit order if not found in data
    if target not in data.data:
        target = target[::-1]

    pair_data = data[target]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {target[0]} - Low Frequency",
            f"Qubit {target[1]} - High Frequency",
        ),
    )
    fitting_report = ""

    fig.add_trace(
        go.Heatmap(
            x=pair_data.length,
            y=pair_data.freq * 10e-6,
            z=data.prob_low,
            coloraxis=COLORAXIS[0],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=pair_data.length,
            y=pair_data.freq * 10e-6,
            z=data.prob_high,
            coloraxis=COLORAXIS[1],
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        xaxis_title="Duration [ns]",
        xaxis2_title="Duration [ns]",
        yaxis_title="Frequency Detuning [MHz]",
        legend=dict(orientation="h"),
    )
    fig.update_layout(
        coloraxis={"colorscale": "Oryel", "colorbar": {"x": 1.15}},
        coloraxis2={"colorscale": "Darkmint", "colorbar": {"x": -0.15}},
    )

    return [fig], fitting_report


def _update(
    results: ChevronCouplersFreqResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    pass


chevron_couplers = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron couplers routine."""
