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
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from .utils import order_pair


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
    nshots: Optional[int] = None
    """Number of shots per point."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""


@dataclass
class ChevronResults(Results):
    """CzFluxTime outputs when fitting will be done."""


ChevronType = np.dtype(
    [("amp", np.float64), ("length", np.float64), ("prob", np.float64)]
)
"""Custom dtype for Chevron."""


@dataclass
class ChevronData(Data):
    """Chevron acquisition outputs."""

    data: dict[tuple[QubitId, QubitId, QubitId], npt.NDArray[ChevronType]] = field(
        default_factory=dict
    )

    def register_qubit(self, low_qubit, high_qubit, qubit, length, amp, prob):
        """Store output for single qubit."""
        size = len(length) * len(amp)
        amplitude, duration = np.meshgrid(amp, length)
        ar = np.empty(size, dtype=ChevronType)
        ar["length"] = duration.ravel()
        ar["amp"] = amplitude.ravel()
        ar["prob"] = prob.ravel()
        self.data[low_qubit, high_qubit, qubit] = np.rec.array(ar)

    def __getitem__(self, pair):
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }


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

        if params.parking:
            for pulse in cz:
                if pulse.qubit not in ordered_pair:
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
        for qubit in ordered_pair:
            result = results[qubit]
            prob = result.magnitude
            data.register_qubit(
                ordered_pair[0],
                ordered_pair[1],
                qubit,
                delta_duration_range,
                delta_amplitude_range,
                prob,
            )
    return data


def _plot(data: ChevronData, fit: ChevronResults, qubit):
    """Plot the experiment result for a single pair."""
    colouraxis = ["coloraxis", "coloraxis2"]
    pair_data = data[qubit]
    # order qubits
    qubits = next(iter(pair_data))[:2]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {qubits[0]} - Low Frequency",
            f"Qubit {qubits[1]} - High Frequency",
        ),
    )
    fit_report = None
    for target, control, measure in pair_data:
        fig.add_trace(
            go.Heatmap(
                x=pair_data[target, control, measure].length,
                y=pair_data[target, control, measure].amp,
                z=pair_data[target, control, measure].prob,
                coloraxis=colouraxis[0 if measure == qubits[0] else 1],
            ),
            row=1,
            col=1 if measure == qubits[0] else 2,
        )

        fig.update_layout(
            xaxis_title="Duration [ns]",
            xaxis2_title="Duration [ns]",
            yaxis_title="Amplitude [dimensionless]",
            legend_title="States",
        )
        fig.update_layout(
            coloraxis={"colorscale": "Oryel", "colorbar": {"x": -0.15}},
            coloraxis2={"colorscale": "Darkmint", "colorbar": {"x": 1.15}},
        )
    return [fig], fit_report


def _fit(data: ChevronData) -> ChevronResults:
    return ChevronResults()


chevron = Routine(_aquisition, _fit, _plot)
"""Chevron routine."""
