"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from ..utils import fit_flux_amplitude, order_pair
from .utils import COLORAXIS, chevron_fit, chevron_sequence


@dataclass
class ChevronParameters(Parameters):
    """CzFluxTime runcard inputs."""

    amplitude_min_factor: float
    """Amplitude minimum."""
    amplitude_max_factor: float
    """Amplitude maximum."""
    amplitude_step_factor: float
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
    native: str = "CZ"
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """

    @property
    def amplitude_range(self):
        return np.arange(
            self.amplitude_min_factor,
            self.amplitude_max_factor,
            self.amplitude_step_factor,
        )

    @property
    def duration_range(self):
        return np.arange(
            self.duration_min,
            self.duration_max,
            self.duration_step,
        )


@dataclass
class ChevronResults(Results):
    """CzFluxTime outputs when fitting will be done."""

    amplitude: dict[QubitPairId, float]
    """CZ angle."""
    duration: dict[QubitPairId, int]
    """Virtual Z phase correction."""
    native: str = "CZ"
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """


ChevronType = np.dtype(
    [
        ("amp", np.float64),
        ("length", np.float64),
        ("prob_high", np.float64),
        ("prob_low", np.float64),
    ]
)
"""Custom dtype for Chevron."""


@dataclass
class ChevronData(Data):
    """Chevron acquisition outputs."""

    native_amplitude: dict[QubitPairId, float] = field(default_factory=dict)
    """CZ platform amplitude for qubit pair."""
    native: str = "CZ"
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """
    sweetspot: dict[QubitPairId, float] = field(default_factory=dict)
    """Sweetspot value for high frequency qubit."""
    data: dict[QubitPairId, npt.NDArray[ChevronType]] = field(default_factory=dict)

    label: Optional[str] = None
    """Label for the data."""

    def register_qubit(self, low_qubit, high_qubit, length, amp, prob_low, prob_high):
        """Store output for single qubit."""
        size = len(length) * len(amp)
        amplitude, duration = np.meshgrid(amp, length)
        ar = np.empty(size, dtype=ChevronType)
        ar["length"] = duration.ravel()
        ar["amp"] = amplitude.ravel()
        ar["prob_low"] = prob_low.ravel()
        ar["prob_high"] = prob_high.ravel()
        self.data[low_qubit, high_qubit] = np.rec.array(ar)

    def amplitudes(self, pair):
        """Unique pair amplitudes."""
        return np.unique(self[pair].amp)

    def durations(self, pair):
        """Unique pair durations."""
        return np.unique(self[pair].length)

    def low_frequency(self, pair):
        return self[pair].prob_low

    def high_frequency(self, pair):
        return self[pair].prob_high


def _aquisition(
    params: ChevronParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> ChevronData:
    r"""Perform an CZ experiment between pairs of qubits by changing its
    frequency.

    Args:
        platform: Platform to use.
        params: Experiment parameters.
        targets (list): List of pairs to use sequentially.

    Returns:
        ChevronData: Acquisition data.
    """

    # create a DataUnits object to store the results
    data = ChevronData(native=params.native)
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        sequence = chevron_sequence(
            platform=platform,
            pair=pair,
            duration_max=params.duration_max,
            parking=params.parking,
            dt=params.dt,
            native=params.native,
        )
        ordered_pair = order_pair(pair, platform)
        # TODO: move in function to avoid code duplications
        sweeper_amplitude = Sweeper(
            Parameter.amplitude,
            params.amplitude_range,
            pulses=[sequence.get_qubit_pulses(ordered_pair[1]).qf_pulses[0]],
            type=SweeperType.FACTOR,
        )
        data.native_amplitude[ordered_pair] = (
            sequence.get_qubit_pulses(ordered_pair[1]).qf_pulses[0].amplitude
        )
        data.sweetspot[ordered_pair] = platform.qubits[ordered_pair[1]].sweetspot
        sweeper_duration = Sweeper(
            Parameter.duration,
            params.duration_range,
            pulses=[sequence.get_qubit_pulses(ordered_pair[1]).qf_pulses[0]],
            type=SweeperType.ABSOLUTE,
        )

        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            sweeper_duration,
            sweeper_amplitude,
        )
        data.register_qubit(
            ordered_pair[0],
            ordered_pair[1],
            params.duration_range,
            params.amplitude_range * data.native_amplitude[ordered_pair],
            results[ordered_pair[0]].probability(state=1),
            results[ordered_pair[1]].probability(state=1),
        )
    return data


def _fit(data: ChevronData) -> ChevronResults:
    durations = {}
    amplitudes = {}
    for pair in data.data:
        amps = data.amplitudes(pair)
        times = data.durations(pair)

        signal = data.low_frequency(pair)
        signal_matrix = signal.reshape(len(times), len(amps)).T

        # guess amplitude computing FFT
        amplitude, index, delta = fit_flux_amplitude(signal_matrix, amps, times)
        # estimate duration by rabi curve at amplitude previously estimated
        y = signal_matrix[index, :].ravel()
        try:
            popt, _ = curve_fit(
                chevron_fit,
                times,
                y,
                p0=[delta * 2 * np.pi, np.pi, np.mean(y), np.mean(y)],
                bounds=(
                    [0, -2 * np.pi, np.min(y), np.min(y)],
                    [np.inf, 2 * np.pi, np.max(y), np.max(y)],
                ),
            )
            # duration can be estimated as the period of the oscillation
            duration = 1 / (popt[0] / 2 / np.pi)
            amplitudes[pair] = amplitude
            durations[pair] = int(duration)
        except Exception as e:
            log.warning(f"Chevron fit failed for pair {pair} due to {e}")

    return ChevronResults(amplitude=amplitudes, duration=durations, native=data.native)


def _plot(data: ChevronData, fit: ChevronResults, target: QubitPairId):
    """Plot the experiment result for a single pair."""
    if isinstance(target, list):
        target = tuple(target)
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

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                target[1],
                [f"{fit.native} amplitude", f"{fit.native} duration", "Bias point"],
                [
                    fit.amplitude[target],
                    fit.duration[target],
                    fit.amplitude[target] + data.sweetspot[target],
                ],
            )
        )

    return [fig], fitting_report


def _update(results: ChevronResults, platform: Platform, target: QubitPairId):
    if isinstance(target, list):
        target = tuple(target)

    if target not in results.duration:
        target = (target[1], target[0])

    getattr(update, f"{results.native}_duration")(
        results.duration[target], platform, target
    )
    getattr(update, f"{results.native}_amplitude")(
        results.amplitude[target], platform, target
    )


chevron = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
