"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import HZ_TO_GHZ, table_dict, table_html

from .... import update
from ..utils import order_pair
from .utils import COLORAXIS, chevron_fit, chevron_sequence

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
    dt: Optional[int] = 0
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""
    native: Literal["CZ", "iSWAP"] = "CZ"
    """Two qubit interaction to be calibrated."""


@dataclass
class ChevronResults(Results):
    """CzFluxTime outputs when fitting will be done."""

    native: str = "CZ"
    """Two qubit interaction to be calibrated."""
    duration: dict[QubitPairId, list] = field(default_factory=dict)
    """Gate duration."""
    half_duration: dict[QubitPairId, list] = field(default_factory=dict)
    """Half gate duration."""
    amplitude: dict[QubitPairId, list] = field(default_factory=dict)
    """Gate flux amplitude."""
    fitted_parameters: dict[QubitPairId, list] = field(default_factory=dict)
    """Fitted parameters for chevron pattern."""


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
    _sorted_pairs: list
    """Pairs to be calibrated sorted by frequency."""
    detuning: dict[QubitPairId, float] = field(default_factory=dict)
    """Expected detuning between qubit in pair."""
    flux_coefficient: dict[QubitPairId, float] = field(default_factory=dict)
    """Flux coefficient to map frequency to amplitude."""
    data: dict[QubitPairId, np.ndarray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def grid(self) -> np.ndarray:
        x, y = np.meshgrid(self.duration, self.amplitude)
        return np.stack([x.ravel(), y.ravel()])

    @property
    def sorted_pairs(self):
        return [
            pair if isinstance(pair, tuple) else tuple(pair)
            for pair in self._sorted_pairs
        ]

    @sorted_pairs.setter
    def sorted_pairs(self, value):
        self._sorted_pairs = value


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
        _sorted_pairs=[order_pair(pair, platform) for pair in targets],
        amplitude=np.arange(
            params.amplitude_min, params.amplitude_max, params.amplitude_step
        ).tolist(),
        duration=np.arange(
            params.duration_min, params.duration_max, params.duration_step
        ).tolist(),
    )
    data.flux_coefficient = {
        pair: platform.calibration.single_qubits[pair[1]].qubit.flux_coefficients[0]
        for pair in data.sorted_pairs
    }
    data.detuning = {
        pair: (
            platform.calibration.single_qubits[pair[1]].qubit.frequency_01
            - platform.calibration.single_qubits[pair[0]].qubit.frequency_01
        )
        * HZ_TO_GHZ
        for pair in data.sorted_pairs
    }
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
    fitted_parameters = {}
    duration = {}
    amplitude = {}
    half_duration = {}
    for pair in data.sorted_pairs:
        fitted_parameters[pair] = []
        duration[pair], half_duration[pair], amplitude[pair] = [], [], []
        for _data in [data[pair][0], data[pair][1]]:
            try:
                popt, _ = curve_fit(
                    chevron_fit,
                    data.grid,
                    _data.T.flatten(),
                    p0=[
                        data.detuning[pair] - 0.2,
                        data.flux_coefficient[pair],
                        0.07,
                        0,
                    ],
                    bounds=(
                        [data.detuning[pair] / 2 - 0.1, -3, 0, 0],
                        [2 * data.detuning[pair] - 0.4, -1, 0.1, 2 * np.pi],
                    ),
                    maxfev=100000,
                )
                fitted_parameters[pair].append(popt.tolist())
                duration[pair].append(np.pi / popt[2])
                half_duration[pair].append(np.pi / popt[2] / 2)
                amplitude[pair].append(np.sqrt(-popt[0] / popt[1]))
            except Exception as e:
                fitted_parameters[pair].append([])
                duration[pair].append([])
                half_duration[pair].append([])
                amplitude[pair].append([])
                log.warning(f"Chevron fit failed for pair {pair} due to {e}")

    return ChevronResults(
        native=data.native,
        fitted_parameters=fitted_parameters,
        duration=duration,
        half_duration=half_duration,
        amplitude=amplitude,
    )


def _plot(data: ChevronData, fit: ChevronResults, target: QubitPairId):
    """Plot the experiment result for a single pair."""

    if target not in data.sorted_pairs:
        target = (target[1], target[0])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Qubit {target[0]} - Low Frequency - Data",
            f"Qubit {target[1]} - High Frequency - Data",
            f"Qubit {target[0]} - Low Frequency - Fit",
            f"Qubit {target[1]} - High Frequency - Fit",
        ),
        shared_xaxes="all",
        shared_yaxes="all",
    )
    fitting_report = ""

    for i, _data in enumerate([data.data[target][0], data.data[target][1]]):
        fig.add_trace(
            go.Heatmap(
                x=data.duration,
                y=data.amplitude,
                z=_data.T,
                coloraxis=COLORAXIS[i],
            ),
            row=1,
            col=1 + i,
        )

    if fit is not None:
        for i in range(2):
            if len(fit.fitted_parameters[target][i]) > 0:
                fig.add_trace(
                    go.Heatmap(
                        x=data.duration,
                        y=data.amplitude,
                        z=chevron_fit(
                            data.grid, *fit.fitted_parameters[target][i]
                        ).reshape(
                            len(data.amplitude),
                            len(data.duration),
                        ),
                        coloraxis=COLORAXIS[i],
                    ),
                    row=2,
                    col=i + 1,
                )

                for j in range(2):
                    fig.add_hline(
                        y=fit.amplitude[target][i],
                        line_dash="dot",
                        row=j + 1,
                        col=i + 1,
                    )

                    fig.add_vline(
                        x=fit.duration[target][i],
                        line_dash="dot",
                        row=j + 1,
                        col=i + 1,
                    )

                    fig.add_vline(
                        x=fit.half_duration[target][i],
                        line_dash="dot",
                        row=j + 1,
                        col=i + 1,
                    )

    fig.update_layout(
        xaxis_title="Duration [ns]",
        xaxis2_title="Duration [ns]",
        xaxis3_title="Duration [ns]",
        xaxis4_title="Duration [ns]",
        yaxis_title="Amplitude [a.u.]",
        yaxis2_title="Amplitude [a.u.]",
        yaxis3_title="Amplitude [a.u.]",
        yaxis4_title="Amplitude [a.u.]",
        legend=dict(orientation="h"),
        coloraxis={"colorscale": "Oryel", "colorbar": {"x": 1.15}},
        coloraxis2={"colorscale": "Darkmint", "colorbar": {"x": -0.15}},
        height=800,
    )
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(showticklabels=True, showline=True, row=i, col=j)
            fig.update_yaxes(showticklabels=True, showline=True, row=i, col=j)

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                target[1],
                [f"{fit.native} amplitude", f"{fit.native} duration"],
                [
                    np.mean(fit.amplitude[target]),
                    np.round(np.mean(fit.duration[target]))
                    if data.native == "CZ"
                    else np.round(np.mean(fit.half_duration[target])),
                ],
            )
        )

    return [fig], fitting_report


def _update(
    results: ChevronResults, platform: CalibrationPlatform, target: QubitPairId
):
    target = target[::-1] if target not in results.duration else target

    getattr(update, f"{results.native}_duration")(
        np.mean(results.duration[target]), platform, target
    )
    getattr(update, f"{results.native}_amplitude")(
        np.mean(results.amplitude[target]), platform, target
    )


chevron = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
