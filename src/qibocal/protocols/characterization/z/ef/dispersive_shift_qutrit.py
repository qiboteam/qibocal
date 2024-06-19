from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Qubits, Results, Routine
from qibocal.protocols.characterization.dispersive_shift import DispersiveShiftType
from qibocal.protocols.characterization.utils import HZ_TO_GHZ, table_dict, table_html
from qibocal.protocols.characterization.z.dispersive_shift import (
    DispersiveShiftData,
    DispersiveShiftParameters,
)


@dataclass
class DispersiveShiftQutritParameters(DispersiveShiftParameters):
    """Dispersive shift inputs."""


@dataclass
class DispersiveShiftQutritResults(Results):
    """Dispersive shift outputs."""

    best_freq: dict[QubitId, float]
    """Readout frequency that maximizes the separation between states in iq-plane"""

    @property
    def state_zero(self):
        return {key: value for key, value in asdict(self).items() if "zero" in key}

    @property
    def state_one(self):
        return {key: value for key, value in asdict(self).items() if "one" in key}

    @property
    def state_two(self):
        return {key: value for key, value in asdict(self).items() if "two" in key}


"""Custom dtype for rabi amplitude."""


@dataclass
class DispersiveShiftQutritData(DispersiveShiftData):
    """Dipsersive shift acquisition outputs."""


def _acquisition(
    params: DispersiveShiftParameters, platform: Platform, qubits: Qubits
) -> DispersiveShiftQutritData:
    r"""
    Data acquisition for dispersive shift experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (DispersiveShiftParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (dict): list of target qubits to perform the action

    """

    # create 3 sequences of pulses for the experiment:
    # sequence_0: I  - I    - MZ
    # sequence_1: RX - I    - MZ
    # sequence_2: RX - RX12 - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    sequence_2 = PulseSequence()

    for qubit in qubits:
        rx_pulse = platform.create_RX_pulse(qubit, start=0)
        rx_12_pulse = platform.create_RX12_pulse(qubit, start=rx_pulse.finish)
        ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence_1.add(rx_pulse)
        sequence_2.add(rx_pulse, rx_12_pulse)
        for sequence in [sequence_0, sequence_1, sequence_2]:
            readout_pulse = deepcopy(ro_pulse)
            # readout_pulse.start = sequence.qd_pulses.finish
            readout_pulse.start = rx_12_pulse.finish
            sequence.add(readout_pulse)
            print(sequence)

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    data = DispersiveShiftQutritData(
        resonator_type=platform.resonator_type, readout_frequency={}
    )

    for state, sequence in enumerate([sequence_0, sequence_1, sequence_2]):
        sweeper = Sweeper(
            Parameter.frequency,
            delta_frequency_range,
            pulses=list(sequence.ro_pulses),
            type=SweeperType.OFFSET,
        )

        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            sweeper,
        )

        for qubit in qubits:
            result = results[qubit]
            data.readout_frequency[qubit] = platform.qubits[
                qubit
            ].native_gates.MZ.frequency
            # store the results
            data.register_qubit(
                DispersiveShiftType,
                (qubit, state),
                dict(
                    freq=sequence.get_qubit_pulses(qubit).ro_pulses[0].frequency
                    + delta_frequency_range,
                    signal=result.magnitude,
                    phase=result.phase,
                    i=result.voltage_i,
                    q=result.voltage_q,
                ),
            )

    return data


def _fit(data: DispersiveShiftQutritData) -> DispersiveShiftQutritResults:
    """Post-Processing for dispersive shift"""
    qubits = data.qubits

    # for each qubit find the frequency that maximizes the area between the states in the iq plane
    best_freqs = {}
    for qubit in qubits:
        x = []
        y = []
        for i in range(3):
            data_i = data[qubit, i]
            x.append(data_i.i)
            y.append(data_i.q)

        frequencies = data[qubit, 0].freq
        areas = 0.5 * np.abs(
            x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
        )
        max_index = np.argmax(areas)
        best_freqs[qubit] = frequencies[max_index]

    return DispersiveShiftQutritResults(best_freq=best_freqs)


def _plot(data: DispersiveShiftQutritData, qubit, fit: DispersiveShiftQutritResults):
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
    data_2 = data[qubit, 2]
    fit_data_0 = fit.state_zero if fit is not None else None
    fit_data_1 = fit.state_one if fit is not None else None
    fit_data_2 = fit.state_two if fit is not None else None
    for i, label, q_data, data_fit in list(
        zip(
            (0, 1, 2),
            ("State 0", "State 1", "State 2"),
            (data_0, data_1, data_2),
            (fit_data_0, fit_data_1, fit_data_2),
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
                    x=[
                        fit.best_freq[qubit] * HZ_TO_GHZ,
                        fit.best_freq[qubit] * HZ_TO_GHZ,
                    ],
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
                    x=[
                        fit.best_freq[qubit] * HZ_TO_GHZ,
                        fit.best_freq[qubit] * HZ_TO_GHZ,
                    ],
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


dispersive_shift_qutrit = Routine(_acquisition, fit=_fit, report=_plot)
