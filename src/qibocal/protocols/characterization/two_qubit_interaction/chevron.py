"""SWAP experiment for two qubit gates, chevron plot."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import Qubit, QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log


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
    nshots: Optional[int] = None
    """Number of shots per point."""


@dataclass
class ChevronResults(Results):
    """CzFluxTime outputs when fitting will be done."""

    # FIXME: update runcard accordingly
    period: dict[str, float]
    """Period of the oscillation"""


ChevronType = np.dtype(
    [("amp", np.float64), ("length", np.float64), ("prob", np.float64)]
)
"""Custom dtype for Chevron."""


@dataclass
class ChevronData(Data):
    """CzFluxTime acquisition outputs."""

    data: dict[tuple[QubitId, QubitId, QubitId], npt.NDArray[ChevronType]] = field(
        default_factory=dict
    )

    def register_qubit(self, low_qubit, high_qubit, qubit, length, amp, prob):
        """Store output for single qubit."""
        size = len(length) * len(amp)
        duration, amplitude = np.meshgrid(length, amp)
        ar = np.empty(size, dtype=ChevronType)
        ar["length"] = duration.ravel()
        ar["amp"] = amplitude.ravel()
        ar["prob"] = prob.ravel()
        self.data[low_qubit, high_qubit, qubit] = np.rec.array(ar)


def order_pairs(
    pair: list[QubitId, QubitId], qubits: dict[QubitId, Qubit]
) -> list[QubitId, QubitId]:
    """Order a pair of qubits by drive frequency."""
    if qubits[pair[0]].drive_frequency > qubits[pair[1]].drive_frequency:
        return pair[::-1]
    return pair


def _aquisition(
    params: ChevronParameters,
    platform: Platform,
    qubits: Qubits,
) -> ChevronData:
    r"""
    Perform an iSWAP/CZ experiment between pairs of qubits by changing its frequency.

    Args:
        platform: Platform to use.
        params: Experiment parameters.
        qubits: Qubits to use.

    Returns:
        DataUnits: Acquisition data.
    """
    if not isinstance(list(qubits.keys())[0], tuple):
        raise ValueError("You need to specify a list of pairs.")

    # create a DataUnits object to store the results,
    data = ChevronData()
    for pair in qubits:
        # order the qubits so that the low frequency one is the first
        sequence = PulseSequence()
        ordered_pair = order_pairs(pair, platform.qubits)

        # initialize in system in 11 state
        for qubit in ordered_pair:
            sequence.add(platform.create_RX_pulse(qubit, start=0))

        # TODO: check if sequence.finish is the correct syntax
        # add flux pulse (ignoring virtual phases)
        cz, _ = platform.create_CZ_pulse_sequence(qubits=pair, start=sequence.finish)
        sequence.add(cz)

        # add readout
        for qubit in ordered_pair:
            sequence.add(
                platform.create_qubit_readout_pulse(qubit, start=sequence.finish)
            )

        # define the parameter to sweep and its range:
        delta_amplitude_range = np.arange(
            params.amplitude_min,
            params.amplitude_max,
            params.amplitude_step,
        )
        delta_duration_range = np.arange(
            params.duration_min, params.duration_max, params.duration_step
        )
        sweeper_amplitude = Sweeper(
            Parameter.amplitude,
            delta_amplitude_range,
            pulses=[cz.get_qubit_pulses(ordered_pair[1])[0]],
            type=SweeperType.FACTOR,
        )
        sweeper_duration = Sweeper(
            Parameter.duration,
            delta_duration_range,
            pulses=[cz[:2]],  # get first two flux pulses
            type=SweeperType.ABSOLUTE,
        )

        # TODO: check if drivers supports duration
        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            sweeper_amplitude,
            sweeper_duration,
        )
        for qubit in ordered_pair:
            result = results[qubit]
            prob = result.probability(1)
            data.register_qubit(
                ordered_pair[0],
                ordered_pair[1],
                qubit,
                delta_duration_range,
                delta_amplitude_range,
                prob,
            )
    return data


def _plot(data: ChevronData, fit: ChevronResults, qubits):
    """Plot the experiment result for a single pair."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("low", "high"))
    states = ["low", "high"]
    # Plot data
    colouraxis = ["coloraxis", "coloraxis2"]

    fit_report = ""
    for state, qubit in zip(states, qubits):
        index = (qubits[0], qubits[1], qubit)
        ordered_index = index if index in data.data else (index[1], index[0], index[2])
        fig.add_trace(
            go.Heatmap(
                x=data[ordered_index].length,
                y=data[ordered_index].amp,
                z=data[ordered_index].prob,
                name=f"Qubit {qubit} |{state}>",
                coloraxis=colouraxis[states.index(state)],
            ),
            row=1,
            col=states.index(state) + 1,
        )

        fit_report += f"q{qubit} - {state} frequency| "
        fit_report += f"Period of oscillation: {fit.period[str(ordered_index)]} ns<br>"

        fig.update_layout(
            title=f"Qubits {qubits[0]}-{qubits[1]} swap frequency",
            xaxis_title="Duration [ns]",
            yaxis_title="Amplitude [dimensionless]",
            legend_title="States",
        )
        fig.update_layout(
            coloraxis={"colorscale": "Oryel", "colorbar": {"x": -0.15}},
            coloraxis2={"colorscale": "Darkmint", "colorbar": {"x": 1.15}},
        )
    return [fig], fit_report


def fit_function(x, p0, p1, p2, p3):
    """Sinusoidal fit function."""
    return p0 + p1 * np.sin(2 * np.pi * p2 * x + p3)


def _fit(data: ChevronData):
    pairs = data.pairs
    results = {}
    for pair in pairs:
        for qubit in pair:
            qubit_data = data[pair[0], pair[1], qubit]
            fft_freqs = []
            for amp in np.unique(qubit_data.amp):
                probability = qubit_data[qubit_data.amp == amp].prob
                fft_freqs.append(max(np.abs(np.fft.fft(probability))))

            min_idx = np.argmin(fft_freqs)
            amp = np.unique(qubit_data.amp)[min_idx]
            duration = qubit_data[qubit_data.amp == amp].length
            probability = qubit_data[qubit_data.amp == amp].amp
            guesses = [np.mean(probability), 1, np.min(fft_freqs), 0]
            # bounds = []
            # TODO maybe normalize
            try:
                popt, _ = curve_fit(
                    fit_function, duration, probability, p0=guesses, maxfev=10000
                )

                results[str((pair[0], pair[1], qubit))] = np.abs(1 / popt[2])

            except:
                log.warning("chevron fit: the fitting was not succesful")

                results[str((pair, qubit))] = 0

    return ChevronResults(results)


chevron = Routine(_aquisition, _fit, _plot)
"""Chevron routine."""
