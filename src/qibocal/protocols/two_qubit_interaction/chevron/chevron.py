"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import (
    DATAFILE,
    Data,
    Parameters,
    Protocol,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from .... import update
from ..utils import order_pair
from .utils import chevron_sequence, fit_gate_parameters

__all__ = ["chevron"]


@dataclass
class ChevronParameters(Parameters):
    """CzFluxTime runcard inputs."""

    amplitude_min: float
    """Amplitude minimum."""
    amplitude_max: float
    """Amplitude maximum."""
    amplitude_step: float
    """Amplitude step."""
    duration_min: float
    """Duration minimum."""
    duration_max: float
    """Duration maximum."""
    duration_step: float
    """Duration step."""
    dt: int | None = 0
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Whether to park non interacting qubits or not."""
    native: Literal["CZ", "iSWAP"] = "CZ"
    """Two qubit interaction to be calibrated."""


@dataclass
class ChevronResults(Results):
    """CzFluxTime outputs when fitting will be done."""

    native: str = "CZ"
    """Two qubit interaction to be calibrated."""
    duration: dict[QubitPairId, float] = field(default_factory=dict)
    """Gate duration [ns], the first return for CZ and the first exchange for iSWAP."""
    amplitude: dict[QubitPairId, float] = field(default_factory=dict)
    """Resonance flux amplitude."""

    def __contains__(self, key):
        # results are stored under the frequency-ordered pair, the report may ask for
        # the pair in runcard order
        return super().__contains__(key) or super().__contains__(tuple(key)[::-1])


@dataclass
class ChevronData(Data):
    """Chevron acquisition outputs."""

    native: str
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """
    amplitude: list
    """Amplitude values."""
    duration: list
    """Duration values."""
    _sorted_pairs: list = field(init=False, repr=False, metadata={"exclude": True})
    """Pairs to be calibrated sorted by frequency."""
    data: dict[QubitPairId, np.ndarray] = field(default_factory=dict)
    """Raw data acquired."""
    label: str | None = None
    """Label for the data."""
    signal: ClassVar[bool] = False
    """Whether ``data`` is readout magnitude instead of probability."""

    @property
    def sorted_pairs(self) -> list:
        return [
            pair if isinstance(pair, tuple) else tuple(pair)
            for pair in self._sorted_pairs
        ]

    @sorted_pairs.setter
    def sorted_pairs(self, value):
        self._sorted_pairs = value

    @classmethod
    def load(cls, path: Path):
        """Generic load method."""
        data_dict = cls.load_data(path, DATAFILE)
        params = cls.load_params(path, DATAFILE)
        pairs = params.pop("_sorted_pairs")
        if data_dict is not None:
            if params is not None:
                instance = cls(data=data_dict, **params)
            else:
                instance = cls(data=data_dict)
        elif params is not None:
            instance = cls(**params)
        instance.sorted_pairs = pairs
        return instance


def _aquisition(
    params: ChevronParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> ChevronData:
    r"""Perform an CZ experiment between pairs of qubits by changing its
    frequency.

    Args:
        platform: CalibrationPlatform to use.
        params: Experiment parameters.
        targets (list): List of pairs to use sequentially.

    Returns:
        ChevronData: Acquisition data.
    """

    data = ChevronData(
        native=params.native,
        amplitude=np.arange(
            params.amplitude_min, params.amplitude_max, params.amplitude_step
        ).tolist(),
        duration=np.arange(
            params.duration_min, params.duration_max, params.duration_step
        ).tolist(),
    )

    data.sorted_pairs = [order_pair(pair, platform) for pair in targets]

    for pair in data.sorted_pairs:
        sequence, flux_pulse, parking_pulses, delays = chevron_sequence(
            platform=platform,
            ordered_pair=pair,
            duration_max=params.duration_max,
            parking=params.parking,
            dt=params.dt,
            native=params.native,
        )
        sweeper_amplitude = Sweeper(
            parameter=Parameter.amplitude,
            range=(params.amplitude_min, params.amplitude_max, params.amplitude_step),
            pulses=[flux_pulse],
        )
        sweeper_duration = Sweeper(
            parameter=Parameter.duration,
            range=(params.duration_min, params.duration_max, params.duration_step),
            pulses=[flux_pulse] + delays + parking_pulses,
        )

        ro_high = list(sequence.channel(platform.qubits[pair[1]].acquisition))[-1]
        ro_low = list(sequence.channel(platform.qubits[pair[0]].acquisition))[-1]

        results = platform.execute(
            [sequence],
            [[sweeper_duration], [sweeper_amplitude]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        data.data[pair[0], pair[1]] = np.stack(
            [
                results[ro_low.id],
                1 - results[ro_high.id]
                if params.native == "CZ"
                else results[ro_high.id],
            ]
        )
    return data


def _fit(data: ChevronData) -> ChevronResults:
    amplitude = {}
    duration = {}
    for pair in data.sorted_pairs:
        try:
            amplitude[pair], duration[pair] = fit_gate_parameters(
                data[pair], data.amplitude, data.duration, data.native, data.signal
            )
        except Exception as e:
            log.warning(f"Chevron fit failed for pair {pair} due to {e}")
    return ChevronResults(native=data.native, amplitude=amplitude, duration=duration)


def _plot(data: ChevronData, fit: ChevronResults, target: QubitPairId):
    """Plot the experiment result for a single pair."""
    # Data is stored under the frequency-ordered pair, but the report may pass
    # the target in the platform's pair order.
    reversed_target = (target[1], target[0])
    data_target = target if target in data.data else reversed_target
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {data_target[0]} - Low Frequency - Data",
            f"Qubit {data_target[1]} - High Frequency - Data",
        ),
        shared_xaxes="all",
    )
    fitting_report = ""
    for i, _data in enumerate([data[data_target][0], data[data_target][1]]):
        fig.add_trace(
            go.Heatmap(
                x=data.duration,
                y=data.amplitude,
                z=_data.T,
                coloraxis="coloraxis",
            ),
            row=1,
            col=1 + i,
        )

    if fit is not None:
        amplitude0, duration0 = fit.amplitude[data_target], fit.duration[data_target]
        x_range = [min(data.duration), max(data.duration)]
        y_range = [min(data.amplitude), max(data.amplitude)]
        lines = {
            "Resonance amplitude": (x_range, [amplitude0] * 2),
            f"{fit.native} duration": ([duration0] * 2, y_range),
        }
        for col in [1, 2]:
            for name, (x, y) in lines.items():
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        # Green is a hue Inferno does not contain.
                        line={"color": "#00c853", "width": 3, "dash": "dash"},
                        name=name,
                        legendgroup=name,
                        showlegend=col == 1,
                    ),
                    row=1,
                    col=col,
                )

    fig.update_layout(
        xaxis_title="Duration [ns]",
        xaxis2_title="Duration [ns]",
        yaxis_title="Amplitude [a.u.]",
        yaxis2_title="Amplitude [a.u.]",
        coloraxis={
            "colorscale": "Inferno",
            # Fix colorbar range to [0, 1]: the plotted values are probabilities.
            "cmin": 0,
            "cmax": 1,
            "colorbar": {"title": {"text": "P<sub>e</sub>"}},
        },
        # Horizontal legend, so it does not overlap the colorbar on the right.
        legend={"orientation": "h", "y": -0.2},
        height=400,
    )
    for j in [1, 2]:
        fig.update_xaxes(showticklabels=True, showline=True, row=1, col=j)
        fig.update_yaxes(showticklabels=True, showline=True, row=1, col=j)

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                data_target[1],
                ["Resonance amplitude", f"{fit.native} duration [ns]"],
                [
                    fit.amplitude[data_target],
                    np.round(fit.duration[data_target], 1),
                ],
            )
        )

    return [fig], fitting_report


def _update(
    results: ChevronResults, platform: CalibrationPlatform, target: QubitPairId
):
    # Fit results use the acquisition order, while native gates use the platform's pair
    # order.
    reversed_target = (target[1], target[0])
    result_target = target if target in results.amplitude else reversed_target
    platform_target = (
        target if target in platform.natives.two_qubit else reversed_target
    )

    getattr(update, f"{results.native}_duration")(
        results.duration[result_target], platform, platform_target
    )
    getattr(update, f"{results.native}_amplitude")(
        results.amplitude[result_target], platform, platform_target
    )


chevron = Protocol(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
