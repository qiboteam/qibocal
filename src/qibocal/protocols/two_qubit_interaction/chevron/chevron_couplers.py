"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from ..utils import order_pair
from .chevron import ChevronData, ChevronParameters
from .utils import COLORAXIS, chevron_sequence

__all__ = ["chevron_couplers"]


@dataclass
class ChevronCouplersParameters(ChevronParameters):
    """ChevronCouplers runcard inputs."""


@dataclass
class ChevronCouplersResults(Results):
    """ChevronCouplers results"""


@dataclass
class ChevronCouplersData(ChevronData):
    """ChevronCouplers acquisition outputs."""


def _aquisition(
    params: ChevronParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> ChevronCouplersData:
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
    data = ChevronData(native=params.native)
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ordered_pair = order_pair(pair, platform)
        sequence, flux_pulse, coupler_pulse, parking_pulses, delays = chevron_sequence(
            platform=platform,
            ordered_pair=ordered_pair,
            duration_max=params.duration_max,
            parking=params.parking,
            dt=params.dt,
            native=params.native,
        )
        sweeper_amplitude = Sweeper(
            parameter=Parameter.amplitude,
            range=(params.amplitude_min, params.amplitude_max, params.amplitude_step),
            pulses=[coupler_pulse],
        )

        sweeper_duration = Sweeper(
            parameter=Parameter.duration,
            range=(params.duration_min, params.duration_max, params.duration_step),
            pulses=[flux_pulse, coupler_pulse] + delays + parking_pulses,
        )

        ro_high = list(sequence.channel(platform.qubits[ordered_pair[1]].acquisition))[
            -1
        ]
        ro_low = list(sequence.channel(platform.qubits[ordered_pair[0]].acquisition))[
            -1
        ]

        data.native_amplitude[ordered_pair] = flux_pulse.amplitude

        results = platform.execute(
            [sequence],
            [[sweeper_duration], [sweeper_amplitude]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        data.register_qubit(
            ordered_pair[0],
            ordered_pair[1],
            sweeper_duration.values,
            sweeper_amplitude.values,
            results[ro_low.id],
            results[ro_high.id],
        )
    return data


def _fit(data: ChevronCouplersData) -> ChevronCouplersResults:
    return ChevronCouplersResults()


def _plot(data: ChevronCouplersData, fit: ChevronCouplersResults, target: QubitPairId):
    """Plot the experiment result for a single pair."""
    # reverse qubit order if not found in data
    if target not in data.data:
        target = (target[1], target[0])

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
            y=pair_data.amp,
            z=data.low_frequency(target),
            coloraxis=COLORAXIS[0],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=pair_data.length,
            y=pair_data.amp,
            z=data.high_frequency(target),
            coloraxis=COLORAXIS[1],
        ),
        row=1,
        col=2,
    )

    for measured_qubit in target:
        if fit is not None:
            fig.add_trace(
                go.Scatter(
                    x=[
                        fit.duration[target],
                    ],
                    y=[
                        fit.amplitude[target],
                    ],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="black",
                        symbol="cross",
                    ),
                    name=f"{data.native} estimate",  #  Change name from the params
                    showlegend=True if measured_qubit == target[0] else False,
                    legendgroup="Voltage",
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        xaxis_title="Duration [ns]",
        xaxis2_title="Duration [ns]",
        yaxis_title=data.label or "Amplitude [a.u.]",
        legend=dict(orientation="h"),
    )
    fig.update_layout(
        coloraxis={"colorscale": "Oryel", "colorbar": {"x": 1.15}},
        coloraxis2={"colorscale": "Darkmint", "colorbar": {"x": -0.15}},
    )

    return [fig], fitting_report


def _update(
    results: ChevronCouplersResults, platform: CalibrationPlatform, target: QubitPairId
):
    pass


chevron_couplers = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron couplers routine."""
