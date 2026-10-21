"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    GaussianSquare,
    Parameter,
    Pulse,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import (
    Data,
    Parameters,
    Protocol,
    QubitId,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    Range,
    RangeLike,
    to_range,
)

__all__ = ["coupler_iswap"]


@dataclass
class CoupleriSWAPParameters(Parameters):
    """iSWAP flux time runcard inputs."""

    coupler_id: QubitPairId
    """Coupler identifier."""
    coupler_amplitude: RangeLike
    """Coupler amplitude range."""
    duration: RangeLike
    """Duration range."""
    dt: float = 0
    """Time delay between flux pulses and readout."""

    @property
    def _coupler_amplitude(self) -> Range:
        return to_range(self.coupler_amplitude)

    @property
    def _duration(self) -> Range:
        return to_range(self.duration)


@dataclass
class CoupleriSWAPResults(Results):
    """iSWAP flux time outputs when fitting will be done."""


@dataclass
class CoupleriSWAPData(Data):
    """iSWAP acquisition outputs."""

    duration: np.ndarray
    """Duration range."""
    amplitude: np.ndarray
    """Amplitude range."""
    data: dict[QubitId, np.ndarray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def grid(self) -> np.ndarray:
        x, y = np.meshgrid(self.duration, self.amplitude)
        return np.stack([x.ravel(), y.ravel()])


def _aquisition(
    params: CoupleriSWAPParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> CoupleriSWAPData:
    r"""Perform an CZ experiment between pairs of qubits by changing its
    frequency.

    Args:
        platform: CalibrationPlatform to use.
        params: Experiment parameters.
        targets (list): List of pairs to use sequentially.

    Returns:
        CoupleriSWAPData: Acquisition data.
    """

    targ = targets[0]

    coupler_pulse = Pulse(
        duration=params._duration[0],
        amplitude=0.0,
        envelope=GaussianSquare(
            risefall=0,
            sigma=0.2,
        ),
        # envelope=Rectangular()
    )

    acq_delay = Delay(duration=params._duration[0])

    q_natives = platform.parameters.native_gates.single_qubit[targ]

    coupler_id = platform.calibration.two_qubits[params.coupler_id].coupler
    if coupler_id is None:
        raise ValueError("Coupler ID should not be None")

    coupler_channel = platform.couplers[coupler_id].flux
    assert coupler_channel is not None, "Coupler channel should not be None"

    q_acq_ch, q_acq_pulse = q_natives.MZ()[0]

    sequence = (
        q_natives.RX()
        | PulseSequence(
            [
                (coupler_channel, coupler_pulse),
                (q_acq_ch, acq_delay),
            ]
        )
        | PulseSequence(
            [
                (q_acq_ch, Delay(duration=params.dt)),
                (q_acq_ch, q_acq_pulse),
            ]
        )
    )

    sweeper_duration = Sweeper(
        parameter=Parameter.duration,
        range=params._duration,
        pulses=[coupler_pulse, acq_delay],
    )

    sweeper_amplitude = Sweeper(
        parameter=Parameter.amplitude,
        range=params._coupler_amplitude,
        pulses=[coupler_pulse],
    )

    results = platform.execute(
        [sequence],
        [[sweeper_duration], [sweeper_amplitude]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = CoupleriSWAPData(
        duration=np.array(sweeper_duration.values).tolist(),
        amplitude=np.array(sweeper_amplitude.values).tolist(),
    )

    data.data[targ] = np.stack(
        [
            results[q_acq_pulse.id],
        ]
    )
    return data


def _fit(data: CoupleriSWAPData) -> CoupleriSWAPResults:

    return CoupleriSWAPResults()


def _plot(data: CoupleriSWAPData, fit: CoupleriSWAPResults, target: QubitId):
    """Plot the experiment result for a single pair."""
    values = np.asarray(data.data[target]).reshape(
        len(data.amplitude), len(data.duration)
    )
    fig = go.Figure(
        go.Heatmap(
            x=data.duration,
            y=data.amplitude,
            z=values,
            colorscale="Inferno",
            colorbar={"title": {"text": "P<sub>e</sub>"}},
        )
    )
    fitting_report = ""

    fig.update_layout(
        title="Qubit-Coupler iSWAP",
        xaxis_title="Duration [ns]",
        yaxis_title="Amplitude [a.u.]",
    )

    return [fig], fitting_report


def _update(
    results: CoupleriSWAPResults, platform: CalibrationPlatform, target: QubitPairId
):
    # Fit results use the acquisition order, while native gates use the platform's pair
    # order.
    return


coupler_iswap = Protocol(_aquisition, _fit, _plot, _update)
"""Chevron routine."""
