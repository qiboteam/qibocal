"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from .chevron.utils import chevron_sequence
from .utils import order_pair

__all__ = ["amplitude_tuning"]


@dataclass
class AmplitudeTuningParameters(Parameters):
    """CzFluxTime runcard inputs."""

    amplitude_coupler_min: float
    """Amplitude minimum."""
    amplitude_coupler_max: float
    """Amplitude maximum."""
    amplitude_coupler_step: float
    """Amplitude step."""
    duration: float
    """Duration minimum."""
    flux_pulse_amplitude: float
    """Duration maximum."""
    dt: Optional[int] = 0
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""
    native: Literal["CZ", "iSWAP"] = "CZ"
    """Two qubit interaction to be calibrated."""

    @property
    def amplitude_coupler_range(self):
        return np.arange(
            self.amplitude_coupler_min,
            self.amplitude_coupler_max,
            self.amplitude_coupler_step,
        ).tolist()


@dataclass
class AmplitudeTuningResults(Results):
    """CzFluxTime outputs when fitting will be done."""


@dataclass
class AmplitudeTuningData(Data):
    """AmplitudeTuning acquisition outputs."""

    native: str
    amplitude_coupler_range: list[float] = field(default_factory=list)
    data: dict[QubitPairId, npt.NDArray] = field(default_factory=dict)


def _aquisition(
    params: AmplitudeTuningParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> AmplitudeTuningData:
    r"""Perform an CZ experiment between pairs of qubits by changing its
    frequency.

    Args:
        platform: CalibrationPlatform to use.
        params: Experiment parameters.
        targets (list): List of pairs to use sequentially.

    Returns:
        AmplitudeTuningData: Acquisition data.
    """

    # create a DataUnits object to store the results
    data = AmplitudeTuningData(
        native=params.native, amplitude_coupler_range=params.amplitude_coupler_range
    )
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ordered_pair = order_pair(pair, platform)
        sequence, flux_pulse, coupler_pulse, parking_pulses, delays = chevron_sequence(
            platform=platform,
            ordered_pair=ordered_pair,
            duration_max=params.duration,
            parking=params.parking,
            dt=params.dt,
            native=params.native,
        )
        sweeper_amplitude = Sweeper(
            parameter=Parameter.amplitude,
            range=(
                params.amplitude_coupler_min,
                params.amplitude_coupler_max,
                params.amplitude_coupler_step,
            ),
            pulses=[coupler_pulse],
        )

        ro_high = list(sequence.channel(platform.qubits[ordered_pair[1]].acquisition))[
            -1
        ]
        ro_low = list(sequence.channel(platform.qubits[ordered_pair[0]].acquisition))[
            -1
        ]

        results = platform.execute(
            [sequence],
            [[sweeper_amplitude]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        data.data[pair] = np.stack([results[ro_low.id], results[ro_high.id]])
    return data


def _fit(data: AmplitudeTuningData) -> AmplitudeTuningResults:
    return AmplitudeTuningResults()


def _plot(data: AmplitudeTuningData, fit: AmplitudeTuningResults, target: QubitPairId):
    """Plot the experiment result for a single pair."""
    # reverse qubit order if not found in data
    if target not in data.data:
        target = (target[1], target[0])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {target[0]} - Low Frequency",
            f"Qubit {target[1]} - High Frequency",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.amplitude_coupler_range,
            y=data.data[target][0],
        ),
        col=1,
        row=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.amplitude_coupler_range,
            y=data.data[target][1],
        ),
        col=2,
        row=1,
    )
    fitting_report = ""

    fig.update_xaxes(title_text="Coupler flux")
    fig.update_yaxes(title_text="Probability")

    return [fig], fitting_report


def _update(
    results: AmplitudeTuningResults, platform: CalibrationPlatform, target: QubitPairId
):
    pass


amplitude_tuning = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""AmplitudeTuning routine."""
