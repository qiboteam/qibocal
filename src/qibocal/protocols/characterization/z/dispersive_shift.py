from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.protocols.characterization.utils import HZ_TO_GHZ, table_dict, table_html


@dataclass
class DispersiveShiftParameters(Parameters):
    """Dispersive shift inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""


@dataclass
class DispersiveShiftResults(Results):
    """Dispersive shift outputs."""

    best_freq: dict[QubitId, float]
    """Readout frequency that maximizes the distance of ground and excited states in iq-plane"""


DispersiveShiftType = np.dtype(
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
class DispersiveShiftData(Data):
    """Dipsersive shift acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    readout_frequency: dict[tuple[QubitId, int]]
    """Current readout frequency."""
    data: dict[tuple[QubitId, int], npt.NDArray[DispersiveShiftType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: DispersiveShiftParameters, platform: Platform, qubits: Qubits
) -> DispersiveShiftData:
    r"""
    Data acquisition for dispersive shift experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (DispersiveShiftParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (dict): list of target qubits to perform the action

    """

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence_0.add(ro_pulses[qubit])
        sequence_1.add(qd_pulses[qubit])
        sequence_1.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    # create a DataUnits objects to store the results
    data = DispersiveShiftData(
        resonator_type=platform.resonator_type, readout_frequency={}
    )
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
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

    # retrieve the results for every qubit
    for qubit in qubits:
        # data.readout_frequency[qubit] = qubits[qubit].readout_frequency
        data.readout_frequency[qubit] = platform.qubits[qubit].native_gates.MZ.frequency
        for i, results in enumerate([results_0, results_1]):
            result = results[ro_pulses[qubit].serial]
            # store the results
            data.register_qubit(
                DispersiveShiftType,
                (qubit, i),
                dict(
                    freq=ro_pulses[qubit].frequency + delta_frequency_range,
                    signal=result.magnitude,
                    phase=result.phase,
                    i=result.voltage_i,
                    q=result.voltage_q,
                ),
            )
    return data


def _fit(data: DispersiveShiftData) -> DispersiveShiftResults:
    """Post-Processing for dispersive shift"""
    qubits = data.qubits
    iq_couples = [[], []]  # axis 0: states, axis 1: qubit

    # for each qubit find the iq couple of 0-1 states that maximize the distance
    best_freqs = {}
    for qubit in qubits:
        iq_couples = []
        for i in range(2):
            data_i = data[qubit, i]
            i_measures = data_i.i
            q_measures = data_i.q
            iq_couples.append(np.stack((i_measures, q_measures), axis=-1))

        frequencies = data[qubit, 0].freq
        max_index = np.argmax(np.linalg.norm(iq_couples[0] - iq_couples[1], axis=-1))
        best_freqs[qubit] = frequencies[max_index]

    return DispersiveShiftResults(best_freq=best_freqs)


def _plot(data: DispersiveShiftData, qubit, fit: DispersiveShiftResults):
    """Plotting function for dispersive shift."""
    figures = []
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
    # iterate over multiple data folders

    fitting_report = ""

    data_0 = data[qubit, 0]
    data_1 = data[qubit, 1]

    for i, label, q_data in list(
        zip(
            (0, 1),
            ("State 0", "State 1"),
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

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=[fit.best_freq[qubit] * HZ_TO_GHZ, fit.best_freq[qubit] * HZ_TO_GHZ],
                y=[
                    np.min(np.concatenate((data_0.signal, data_1.signal))),
                    np.max(np.concatenate((data_0.signal, data_1.signal))),
                ],
                mode="lines",
                line=go.scatter.Line(color="orange", width=2, dash="dash"),
                name="Best frequency",
                showlegend=True,
                legendgroup="Best frequency",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[fit.best_freq[qubit] * HZ_TO_GHZ, fit.best_freq[qubit] * HZ_TO_GHZ],
                y=[-np.pi, np.pi],
                mode="lines",
                line=go.scatter.Line(color="orange", width=2, dash="dash"),
                name="Best frequency",
                showlegend=False,
                legendgroup="Best frequency",
            ),
            row=1,
            col=2,
        )

        fitting_report = table_html(
            table_dict(
                qubit,
                [
                    "Best Frequency [Hz]",
                ],
                np.round(
                    [
                        fit.best_freq[qubit],
                    ]
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=np.abs((data_0.i + 1j * data_0.q) - (data_1.i + 1j * data_1.q)),
            opacity=opacity,
            name=f"Distance iq plane",
            showlegend=True,
            legendgroup=f"Distance",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=np.abs(data_0.signal - data_1.signal),
            opacity=opacity,
            name=f"Magnitude difference",
            showlegend=True,
            legendgroup=f"Distance",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[
                data.readout_frequency[qubit] * HZ_TO_GHZ,
                data.readout_frequency[qubit] * HZ_TO_GHZ,
            ],
            y=[
                np.min(np.concatenate((data_0.signal, data_1.signal))),
                np.max(np.concatenate((data_0.signal, data_1.signal))),
            ],
            mode="lines",
            line=go.scatter.Line(color="black", width=2),
            name="Current frequency",
            showlegend=True,
            legendgroup="Current frequency",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[
                data.readout_frequency[qubit] * HZ_TO_GHZ,
                data.readout_frequency[qubit] * HZ_TO_GHZ,
            ],
            y=[-np.pi, np.pi],
            mode="lines",
            line=go.scatter.Line(color="black", width=2),
            name="Current frequency",
            showlegend=False,
            legendgroup="Current frequency",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHz]",
        yaxis_title="Signal [a.u.]",
        xaxis2_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: DispersiveShiftResults, platform: Platform, qubit: QubitId):
    update.readout_frequency(results.best_freq[qubit], platform, qubit)


dispersive_shift = Routine(_acquisition, _fit, _plot, _update)
"""Dispersive shift Routine object."""
