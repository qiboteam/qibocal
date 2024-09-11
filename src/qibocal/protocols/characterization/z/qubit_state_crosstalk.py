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
from qibocal.protocols.characterization.utils import HZ_TO_GHZ


@dataclass
class CrosstalkParameters(Parameters):
    """Dispersive shift inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    drive_duration: int
    """Drive pulse duration [ns]. Same for all qubits."""
    drive_amplitude: Optional[float] = None
    """Drive pulse amplitude (optional). Same for all qubits."""

    excited_qubit: Optional[QubitId] = None
    affected_qubit: Optional[QubitId] = None
    delay_before_spectroscopy: Optional[int] = 16


@dataclass
class CrosstalkResults(Results):
    """Dispersive shift outputs."""

    best_freq: dict[QubitId, float]
    """Readout frequency that maximizes the distance of ground and excited states in iq-plane"""


CrosstalkType = np.dtype(
    [
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class CrosstalkData(Data):
    """Dipsersive shift acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    qubit_frequency: dict[tuple[QubitId, int]]
    """Current readout frequency."""
    data: dict[tuple[QubitId, int], npt.NDArray[CrosstalkType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: CrosstalkParameters, platform: Platform, qubits: Qubits
) -> CrosstalkData:
    r"""
    Data acquisition for qubit state crosstalk experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (CrosstalkParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (dict): list of target qubits to perform the action

    """

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()

    # define pulses
    excitation_pulse = platform.create_RX_pulse(params.excited_qubit, start=0)
    spectroscopy_pulse = platform.create_qubit_drive_pulse(
        params.affected_qubit,
        start=excitation_pulse.finish + params.delay_before_spectroscopy,
        duration=params.drive_duration,
    )
    if params.drive_amplitude is not None:
        spectroscopy_pulse.amplitude = params.drive_amplitude

    readout_pulse = platform.create_qubit_readout_pulse(
        params.affected_qubit, start=spectroscopy_pulse.finish
    )

    # add pulses to pulse sequences
    sequence_0.add(spectroscopy_pulse, readout_pulse)
    sequence_1.add(excitation_pulse, spectroscopy_pulse, readout_pulse)

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    # create a DataUnits objects to store the results
    data = CrosstalkData(resonator_type=platform.resonator_type, qubit_frequency={})
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[spectroscopy_pulse],
        type=SweeperType.OFFSET,
    )

    results_0 = platform.sweep(
        sequence_0,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    results_1 = platform.sweep(
        sequence_1,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    # data.readout_frequency[qubit] = qubits[qubit].readout_frequency
    data.qubit_frequency[params.affected_qubit] = platform.qubits[
        params.affected_qubit
    ].native_gates.MZ.frequency
    for i, results in enumerate([results_0, results_1]):
        result = results[readout_pulse.serial]
        # store the results
        data.register_qubit(
            CrosstalkType,
            (params.affected_qubit, i),
            dict(
                freq=spectroscopy_pulse.frequency + delta_frequency_range,
                signal=result.magnitude,
                phase=result.phase,
                i=result.voltage_i,
                q=result.voltage_q,
            ),
        )
    return data


def _fit(data: CrosstalkData) -> CrosstalkResults:
    """Post-Processing for qubit state crosstalk"""
    best_freqs = {}

    return CrosstalkResults(best_freq=best_freqs)


def _plot(data: CrosstalkData, qubit, fit: CrosstalkResults):
    """Plotting function for qubit state crosstalk."""
    figures = []
    fitting_report = ""
    # iterate over multiple data folders

    if (qubit, 0) in data.data:
        fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=(
                "Signal [a.u.]",
                "phase [rad]",
            ),
        )
        data_0 = data[qubit, 0]
        data_1 = data[qubit, 1]

        for i, label, q_data in list(
            zip(
                (0, 1),
                ("Excited qubit in state 0", "Excited qubit in state 1"),
                (data_0, data_1),
            )
        ):
            opacity = 1
            frequencies = q_data.freq * HZ_TO_GHZ
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=q_data.signal,
                    opacity=opacity,
                    name=f"{label}",
                    showlegend=True,
                    legendgroup=f"{label}",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=q_data.phase,
                    opacity=opacity,
                    showlegend=False,
                    legendgroup=f"{label}",
                ),
                row=1,
                col=2,
            )

        fig.update_layout(
            showlegend=True,
            xaxis_title="Qubit Frequency [GHz]",
            yaxis_title="Signal [a.u.]",
            xaxis2_title="Qubit Frequency [GHz]",
            yaxis2_title="Phase [rad]",
        )

        figures.append(fig)

    return figures, fitting_report


def _update(results: CrosstalkResults, platform: Platform, qubit: QubitId):
    pass


qubit_state_crosstalk = Routine(_acquisition, _fit, _plot, _update)
"""Qubit state crosstalk experiment"""
