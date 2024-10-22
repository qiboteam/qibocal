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
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    lorentzian,
    lorentzian_fit,
    table_dict,
    table_html,
)


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

    frequencies: dict[QubitId, list[float]]
    """Qubit peak frequencies."""
    fitted_parameters: dict[QubitId, list[list[float]]]
    """Fitted parameters. The first element is the resonator frequency when the
    qubit is in the ground state, the second one when the qubit is in the first excited
    state."""
    best_freq: dict[QubitId, float]
    """Readout frequency that maximizes the distance of ground and excited states in iq-plane"""

    def chi(self, target: QubitId) -> float:
        "Evaluate the dispersive shift"
        freq = self.frequencies[target]
        return (freq[0] - freq[1]) / 2


DispersiveShiftType = np.dtype(
    [
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for dispersive shift."""


@dataclass
class DispersiveShiftData(Data):
    """Dispersive shift acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[tuple[QubitId, int], npt.NDArray[DispersiveShiftType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: DispersiveShiftParameters, platform: Platform, targets: list[QubitId]
) -> DispersiveShiftData:
    r"""
    Data acquisition for dispersive shift experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (DispersiveShiftParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        targets (list): list of target qubits to perform the action

    """

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in targets:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence_0.add(ro_pulses[qubit])
        sequence_1.add(qd_pulses[qubit])
        sequence_1.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    # create a DataUnits objects to store the results
    data = DispersiveShiftData(resonator_type=platform.resonator_type)
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in targets],
        type=SweeperType.OFFSET,
    )

    execution_pars = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    results_0 = platform.sweep(
        sequence_0,
        execution_pars,
        sweeper,
    )

    results_1 = platform.sweep(
        sequence_1,
        execution_pars,
        sweeper,
    )

    # retrieve the results for every qubit
    for qubit in targets:
        for i, results in enumerate([results_0, results_1]):
            result = results[ro_pulses[qubit].serial].average
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

    res_frequencies = {}
    best_freqs = {}
    fitted_parameters = {}

    for qubit in qubits:
        freq = []
        fit_params = []
        for i in range(2):
            data_i = data[qubit, i]
            fit_result = lorentzian_fit(
                data_i, resonator_type=data.resonator_type, fit="resonator"
            )
            if fit_result is None:
                freq = fit_params = None
                break

            freq.append(fit_result[0])
            fit_params.append(fit_result[1])

        res_frequencies[qubit] = freq
        fitted_parameters[qubit] = fit_params

    for idx, qubit in enumerate(qubits):
        for i in range(2):
            data_i = data[qubit, i]
            i_measures = data_i.i
            q_measures = data_i.q
            iq_couples[i].append(np.stack((i_measures, q_measures), axis=-1))

        frequencies = data[qubit, 0].freq
        max_index = np.argmax(
            np.linalg.norm(iq_couples[0][idx] - iq_couples[1][idx], axis=-1)
        )
        best_freqs[qubit] = frequencies[max_index]

    return DispersiveShiftResults(
        frequencies=res_frequencies,
        fitted_parameters=fitted_parameters,
        best_freq=best_freqs,
    )


def _plot(data: DispersiveShiftData, target: QubitId, fit: DispersiveShiftResults):
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

    fitting_report = ""

    data_0 = data[target, 0]
    data_1 = data[target, 1]

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
            fig.add_vline(
                x=fit.best_freq[target] * HZ_TO_GHZ,
                line=dict(color="orange", width=3, dash="dash"),
                row=1,
                col=1,
            )
            table_entries = [
                "Best Frequency [Hz]",
            ]
            table_values = np.round(
                [
                    fit.best_freq[target],
                ]
            )

            if fit.frequencies[target] is not None:
                freqrange = np.linspace(
                    min(frequencies),
                    max(frequencies),
                    2 * len(q_data),
                )
                params = fit.fitted_parameters[target][i]
                fig.add_trace(
                    go.Scatter(
                        x=freqrange,
                        y=lorentzian(freqrange, *params),
                        name=f"{label} Fit",
                        line=go.scatter.Line(dash="dot"),
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=[
                            fit.best_freq[target] * HZ_TO_GHZ,
                            fit.best_freq[target] * HZ_TO_GHZ,
                        ],
                        y=[
                            np.min(np.concatenate((data_0.signal, data_1.signal))),
                            np.max(np.concatenate((data_0.signal, data_1.signal))),
                        ],
                        mode="lines",
                        line=go.scatter.Line(color="orange", width=3, dash="dash"),
                        name="Best frequency",
                    ),
                    row=1,
                    col=1,
                )
                table_entries = [
                    "State Zero Frequency [Hz]",
                    "State One Frequency [Hz]",
                    "Chi [Hz]",
                    "Best Frequency [Hz]",
                ]
                table_values = np.round(
                    [
                        fit.frequencies[target][0],
                        fit.frequencies[target][1],
                        fit.chi(target),
                        fit.best_freq[target],
                    ]
                )

            fitting_report = table_html(table_dict(target, table_entries, table_values))
    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHz]",
        yaxis_title="Signal [a.u.]",
        xaxis2_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: DispersiveShiftResults, platform: Platform, target: QubitId):
    update.readout_frequency(results.best_freq[target], platform, target)
    if results.frequencies[target] is not None:
        delta = platform.qubits[target].drive_frequency - results.frequencies[target][0]
        g = np.sqrt(np.abs(results.chi(target) * delta))
        update.coupling(g, platform, target)


dispersive_shift = Routine(_acquisition, _fit, _plot, _update)
"""Dispersive shift Routine object."""
