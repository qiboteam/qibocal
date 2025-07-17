"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from ..utils import order_pair
from .utils import chevron_sequence

__all__ = ["coupler_amplitude"]


@dataclass
class CouplerAmplitudeParameters(Parameters):
    """CzFluxTime runcard inputs."""

    amplitude_min: float
    """Amplitude minimum."""
    amplitude_max: float
    """Amplitude maximum."""
    amplitude_step: float
    """Amplitude step."""
    amplitude_coupler_min: float
    """Amplitude coupler minimum."""
    amplitude_coupler_max: float
    """Amplitude coupler maximum."""
    amplitude_coupler_step: float
    """Amplitude coupler step."""
    duration: int
    """Duration of coupler and qubit flux pulses"""
    dt: Optional[int] = 0
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""
    native: Literal["CZ", "iSWAP"] = "CZ"
    """Two qubit interaction to be calibrated."""

    @property
    def amplitude_range(self) -> list:
        return np.arange(
            self.amplitude_min, self.amplitude_max, self.amplitude_step
        ).tolist()

    @property
    def amplitude_coupler_range(self) -> list:
        return np.arange(
            self.amplitude_coupler_min,
            self.amplitude_coupler_max,
            self.amplitude_coupler_step,
        ).tolist()


@dataclass
class CouplerAmplitudeResults(Results):
    """CzFluxTime outputs when fitting will be done."""


CouplerAmplitudeType = np.dtype(
    [
        ("amp", np.float64),
        ("length", np.float64),
        ("prob_high", np.float64),
        ("prob_low", np.float64),
    ]
)
"""Custom dtype for CouplerAmplitude."""


@dataclass
class CouplerAmplitudeData(Data):
    """CouplerAmplitude acquisition outputs."""

    amplitude_range: list
    """Flux pulse amplitude range for qubit."""
    amplitude_coupler_range: list
    """Flux pulse amplitude range for coupler."""
    native: str = "CZ"
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """
    data: dict[QubitPairId, npt.NDArray] = field(default_factory=dict)


def _aquisition(
    params: CouplerAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerAmplitudeData:
    r"""Perform an CZ experiment between pairs of qubits by changing its
    frequency.

    Args:
        platform: CalibrationPlatform to use.
        params: Experiment parameters.
        targets (list): List of pairs to use sequentially.

    Returns:
        CouplerAmplitudeData: Acquisition data.
    """

    data = CouplerAmplitudeData(
        native=params.native,
        amplitude_range=params.amplitude_range,
        amplitude_coupler_range=params.amplitude_coupler_range,
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
            range=(params.amplitude_min, params.amplitude_max, params.amplitude_step),
            pulses=[flux_pulse],
        )

        sweeper_amplitude_coupler = Sweeper(
            parameter=Parameter.amplitude,
            range=(
                params.amplitude_coupler_min,
                params.amplitude_coupler_max,
                params.amplitude_coupler_step,
            ),
            pulses=[coupler_pulse],
        )
        # ro_high = list(sequence.channel(platform.qubits[ordered_pair[1]].acquisition))[
        #     -1
        # ]
        ro_low = list(sequence.channel(platform.qubits[ordered_pair[0]].acquisition))[
            -1
        ]

        results = platform.execute(
            [sequence],
            [[sweeper_amplitude], [sweeper_amplitude_coupler]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        data.data[pair] = results[ro_low.id]

    return data


def _fit(data: CouplerAmplitudeData) -> CouplerAmplitudeResults:
    return CouplerAmplitudeResults()


def _plot(
    data: CouplerAmplitudeData, fit: CouplerAmplitudeResults, target: QubitPairId
):
    """Plot the experiment result for a single pair."""

    figures = []
    fig = go.Figure()
    fitting_report = ""
    fig.add_trace(
        go.Heatmap(
            x=data.amplitude_range,
            y=data.amplitude_coupler_range,
            z=data.data[target].T,
        ),
    )
    fig.update_xaxes(title_text="Qubit flux")
    fig.update_yaxes(title_text="Coupler flux")

    fig.update_layout(
        showlegend=False,
    )

    figures.append(fig)
    return figures, fitting_report


def _update(
    results: CouplerAmplitudeResults, platform: CalibrationPlatform, target: QubitPairId
):
    pass


coupler_amplitude = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""CouplerAmplitude routine."""
