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
from qibolab.qubits import QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.protocols.characterization.utils import table_dict, table_html

from .utils import fit_flux_amplitude, order_pair

COLORAXIS = ["coloraxis2", "coloraxis1"]


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


@dataclass
class ChevronResults(Results):
    """CzFluxTime outputs when fitting will be done."""

    amplitude: dict[QubitPairId, float]
    """CZ angle."""
    duration: dict[QubitPairId, int]
    """Virtual Z phase correction."""


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

    data: dict[QubitPairId, npt.NDArray[ChevronType]] = field(default_factory=dict)

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
        ChevronData: Acquisition data.
    """

    # create a DataUnits object to store the results,
    data = ChevronData()
    for pair in qubits:
        # order the qubits so that the low frequency one is the first
        sequence = PulseSequence()
        ordered_pair = order_pair(pair, platform.qubits)
        # initialize in system in 11 state
        initialize_lowfreq = platform.create_RX_pulse(
            ordered_pair[0], start=0, relative_phase=0
        )
        initialize_highfreq = platform.create_RX_pulse(
            ordered_pair[1], start=0, relative_phase=0
        )
        sequence.add(initialize_highfreq)
        sequence.add(initialize_lowfreq)
        cz, _ = platform.create_CZ_pulse_sequence(
            qubits=(ordered_pair[1], ordered_pair[0]),
            start=initialize_highfreq.finish,
        )

        sequence.add(cz.get_qubit_pulses(ordered_pair[0]))
        sequence.add(cz.get_qubit_pulses(ordered_pair[1]))

        # Patch to get the coupler until the routines use QubitPair
        if platform.couplers:
            sequence.add(
                cz.coupler_pulses(
                    platform.pairs[tuple(sorted(ordered_pair))].coupler.name
                )
            )

        if params.parking:
            for pulse in cz:
                if pulse.qubit not in ordered_pair:
                    pulse.start = 0
                    pulse.duration = 100
                    sequence.add(pulse)

        # add readout
        measure_lowfreq = platform.create_qubit_readout_pulse(
            ordered_pair[0],
            start=initialize_lowfreq.finish + params.duration_max + params.dt,
        )
        measure_highfreq = platform.create_qubit_readout_pulse(
            ordered_pair[1],
            start=initialize_highfreq.finish + params.duration_max + params.dt,
        )

        sequence.add(measure_lowfreq)
        sequence.add(measure_highfreq)

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
            pulses=[cz.get_qubit_pulses(ordered_pair[1]).qf_pulses[0]],
            type=SweeperType.ABSOLUTE,
        )
        sweeper_duration = Sweeper(
            Parameter.duration,
            delta_duration_range,
            pulses=[cz.get_qubit_pulses(ordered_pair[1]).qf_pulses[0]],
            type=SweeperType.ABSOLUTE,
        )
        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            sweeper_duration,
            sweeper_amplitude,
        )
        data.register_qubit(
            ordered_pair[0],
            ordered_pair[1],
            delta_duration_range,
            delta_amplitude_range,
            results[ordered_pair[0]].magnitude,
            results[ordered_pair[1]].magnitude,
        )
    return data


# fitting function for single row in chevron plot (rabi-like curve)
def cos(x, omega, phase, amplitude, offset):
    return amplitude * np.cos(x * omega + phase) + offset


def _fit(data: ChevronData) -> ChevronResults:
    durations = {}
    amplitudes = {}
    for pair in data.data:
        pair_amplitude = []
        pair_duration = []
        amps = np.unique(data[pair].amp)
        times = np.unique(data[pair].length)

        for qubit in pair:
            msr = data[pair].prob_low if pair[0] == qubit else data[pair].prob_high
            msr_matrix = msr.reshape(len(times), len(amps)).T

            # guess amplitude computing FFT
            amplitude, index, delta = fit_flux_amplitude(msr_matrix, amps, times)
            # estimate duration by rabi curve at amplitude previously estimated
            y = msr_matrix[index, :].ravel()

            popt, _ = curve_fit(cos, times, y, p0=[delta, 0, np.mean(y), np.mean(y)])

            # duration can be estimated as the period of the oscillation
            duration = 1 / (popt[0] / 2 / np.pi)
            pair_amplitude.append(amplitude)
            pair_duration.append(duration)

        amplitudes[pair] = np.mean(pair_amplitude)
        durations[pair] = int(np.mean(duration))

    return ChevronResults(amplitude=amplitudes, duration=durations)


def _plot(data: ChevronData, fit: ChevronResults, qubit):
    """Plot the experiment result for a single pair."""

    # reverse qubit order if not found in data
    if qubit not in data.data:
        qubit = (qubit[1], qubit[0])

    pair_data = data[qubit]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {qubit[0]} - Low Frequency",
            f"Qubit {qubit[1]} - High Frequency",
        ),
    )
    fitting_report = ""

    fig.add_trace(
        go.Heatmap(
            x=pair_data.length,
            y=pair_data.amp,
            z=pair_data.prob_low,
            coloraxis=COLORAXIS[0],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=pair_data.length,
            y=pair_data.amp,
            z=pair_data.prob_high,
            coloraxis=COLORAXIS[1],
        ),
        row=1,
        col=2,
    )

    for measured_qubit in qubit:
        if fit is not None:
            fig.add_trace(
                go.Scatter(
                    x=[
                        fit.duration[qubit],
                    ],
                    y=[
                        fit.amplitude[qubit],
                    ],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="black",
                        symbol="cross",
                    ),
                    name="CZ estimate",
                    showlegend=True if measured_qubit == qubit[0] else False,
                    legendgroup="Voltage",
                ),
                row=1,
                col=1 if measured_qubit == qubit[0] else 2,
            )

    fig.update_layout(
        xaxis_title="Duration [ns]",
        xaxis2_title="Duration [ns]",
        yaxis_title="Amplitude [dimensionless]",
        legend=dict(orientation="h"),
    )
    fig.update_layout(
        coloraxis={"colorscale": "Oryel", "colorbar": {"x": 1.15}},
        coloraxis2={"colorscale": "Darkmint", "colorbar": {"x": -0.15}},
    )

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                qubit[1],
                ["CZ amplitude", "CZ duration"],
                [fit.amplitude[qubit], fit.duration[qubit]],
            )
        )

    return [fig], fitting_report


def _update(results: ChevronResults, platform: Platform, qubit_pair: QubitPairId):
    if qubit_pair not in results.duration:
        qubit_pair = (qubit_pair[1], qubit_pair[0])
    update.CZ_duration(results.duration[qubit_pair], platform, qubit_pair)
    update.CZ_amplitude(results.amplitude[qubit_pair], platform, qubit_pair)


chevron = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
