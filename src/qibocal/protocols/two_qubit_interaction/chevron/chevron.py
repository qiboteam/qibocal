"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from .... import update
from ..utils import fit_flux_amplitude, order_pair
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

    amplitude: dict[QubitPairId, float]
    """CZ angle."""
    duration: dict[QubitPairId, int]
    """Virtual Z phase correction."""
    native: Literal["CZ", "iSWAP"] = "CZ"
    """Two qubit interaction to be calibrated."""


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
        # Since an X gate was added on the high frequency qubit at the end of the
        # pulse sequence, its Chevron pattern is between state 0 and 2, so the state
        # one is mapped into 0. For this reason and compatibility with the other
        # qubit, we have to evaluate the ground state probability.
        ar["prob_high"] = 1 - prob_high.ravel()
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

    # create a DataUnits object to store the results
    data = ChevronData(native=params.native)
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ordered_pair = order_pair(pair, platform)
        sequence, flux_pulse, parking_pulses, delays = chevron_sequence(
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
            pulses=[flux_pulse],
        )
        sweeper_duration = Sweeper(
            parameter=Parameter.duration,
            range=(params.duration_min, params.duration_max, params.duration_step),
            pulses=[flux_pulse] + delays + parking_pulses,
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
            period = 1 / (popt[0] / 2 / np.pi)
            duration = period if data.native == "CZ" else period / 2
            amplitudes[pair] = amplitude
            durations[pair] = int(duration)
        except Exception as e:
            log.warning(f"Chevron fit failed for pair {pair} due to {e}")

    return ChevronResults(amplitude=amplitudes, duration=durations, native=data.native)


def _plot(data: ChevronData, fit: ChevronResults, target: QubitPairId):
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

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                target[1],
                [f"{fit.native} amplitude", f"{fit.native} duration"],
                [
                    fit.amplitude[target],
                    fit.duration[target],
                ],
            )
        )

    return [fig], fitting_report


def _update(
    results: ChevronResults, platform: CalibrationPlatform, target: QubitPairId
):
    target = target[::-1] if target not in results.duration else target

    getattr(update, f"{results.native}_duration")(
        results.duration[target], platform, target
    )
    getattr(update, f"{results.native}_amplitude")(
        results.amplitude[target], platform, target
    )


chevron = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
