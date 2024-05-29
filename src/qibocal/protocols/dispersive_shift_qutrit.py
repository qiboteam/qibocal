from dataclasses import asdict, dataclass, field

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
from qibocal.protocols.dispersive_shift import DispersiveShiftType
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    lorentzian,
    lorentzian_fit,
    table_dict,
    table_html,
)


@dataclass
class DispersiveShiftQutritParameters(Parameters):
    """Dispersive shift inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""


@dataclass
class DispersiveShiftQutritResults(Results):
    """Dispersive shift outputs."""

    frequency_state_zero: dict[QubitId, float]
    """State zero frequency."""
    frequency_state_one: dict[QubitId, float]
    """State one frequency."""
    frequency_state_two: dict[QubitId, float]
    """State two frequency."""
    fitted_parameters_state_zero: dict[QubitId, list[float]]
    """Fitted parameters state zero."""
    fitted_parameters_state_one: dict[QubitId, list[float]]
    """Fitted parameters state one."""
    fitted_parameters_state_two: dict[QubitId, list[float]]
    """Fitted parameters state one."""
    best_freqs01: dict[QubitId, float]
    """Best readout frequency."""
    best_freqs12: dict[QubitId, float]
    """Best readout frequency."""

    @property
    def state_zero(self):
        return {key: value for key, value in asdict(self).items() if "zero" in key}

    @property
    def state_one(self):
        return {key: value for key, value in asdict(self).items() if "one" in key}

    @property
    def state_two(self):
        return {key: value for key, value in asdict(self).items() if "two" in key}


@dataclass
class DispersiveShiftQutritData(Data):
    """Dispersive shift acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[tuple[QubitId, int], npt.NDArray[DispersiveShiftType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: DispersiveShiftQutritParameters, platform: Platform, targets: list[QubitId]
) -> DispersiveShiftQutritData:
    r"""
    Data acquisition for dispersive shift experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (DispersiveShiftQutritParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        targets (list): list of target qubits to perform the action

    """

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    sequence_2 = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    rx12_pulses = {}
    for qubit in targets:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = []
        rx12_pulses[qubit] = platform.create_RX12_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        ro_pulses[qubit].append(platform.create_qubit_readout_pulse(qubit, start=0))

        ro_pulses[qubit].append(
            platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].duration)
        )
        ro_pulses[qubit].append(
            platform.create_qubit_readout_pulse(
                qubit, start=rx12_pulses[qubit].duration
            )
        )
        sequence_0.add(ro_pulses[qubit][0])

        sequence_1.add(qd_pulses[qubit])
        sequence_1.add(ro_pulses[qubit][1])

        sequence_2.add(qd_pulses[qubit])
        sequence_2.add(rx12_pulses[qubit])
        sequence_2.add(ro_pulses[qubit][2])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    data = DispersiveShiftQutritData(resonator_type=platform.resonator_type)
    results = []
    for state, sequence in enumerate([sequence_0, sequence_1, sequence_2]):
        sweeper = Sweeper(
            Parameter.frequency,
            delta_frequency_range,
            pulses=[ro_pulses[qubit][state] for qubit in targets],
            type=SweeperType.OFFSET,
        )

        results.append(
            platform.sweep(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
                sweeper,
            )
        )

    # retrieve the results for every qubit
    for qubit in targets:
        for state, results in enumerate(results):
            result = results[ro_pulses[qubit][state].serial]
            # store the results
            data.register_qubit(
                DispersiveShiftType,
                (qubit, state),
                dict(
                    freq=ro_pulses[qubit][state].frequency + delta_frequency_range,
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
    iq_couples = [[], []]  # axis 0: states, axis 1: qubit

    frequency_0 = {}
    frequency_1 = {}
    frequency_2 = {}
    fitted_parameters_0 = {}
    fitted_parameters_1 = {}
    fitted_parameters_2 = {}
    best_freqs01 = {}
    best_freqs12 = {}
    frequencies = np.unique(data[0, 0].freq)
    for qubit in qubits:
        centers = [[] for _ in range(3)]
        for state in range(3):
            data_state = data[qubit, state]
            # print(data_state)
            fit_result = lorentzian_fit(
                data_state, resonator_type=data.resonator_type, fit="resonator"
            )
            if fit_result is not None:
                if state == 0:
                    frequency_0[qubit], fitted_parameters_0[qubit], _ = fit_result
                elif state == 1:
                    frequency_1[qubit], fitted_parameters_1[qubit], _ = fit_result
                else:
                    frequency_2[qubit], fitted_parameters_2[qubit], _ = fit_result
            for frequency in frequencies:
                data_same_freq = data_state[data_state.freq == frequency]
                i_center = np.mean(data_same_freq.i)
                q_center = np.mean(data_same_freq.q)
                centers[state].append([i_center, q_center])
        centers = np.array(centers)
        # import pdb; pdb.set_trace()
        distances01 = np.linalg.norm(centers[0] - centers[1], axis=1)
        distances12 = np.linalg.norm(centers[1] - centers[2], axis=1)
        # distances = np.stack((distances01, distances12))
        # distances = np.sum(distances, axis=0)
        best_freqs01[qubit] = frequencies[np.argmax(distances01)]
        best_freqs12[qubit] = frequencies[np.argmax(distances12)]
    return DispersiveShiftQutritResults(
        frequency_state_zero=frequency_0,
        frequency_state_one=frequency_1,
        frequency_state_two=frequency_2,
        fitted_parameters_state_one=fitted_parameters_1,
        fitted_parameters_state_zero=fitted_parameters_0,
        fitted_parameters_state_two=fitted_parameters_2,
        best_freqs01=best_freqs01,
        best_freqs12=best_freqs12,
    )


def _plot(
    data: DispersiveShiftQutritData, target: QubitId, fit: DispersiveShiftQutritResults
):
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

    data_0 = data[target, 0]
    data_1 = data[target, 1]
    data_2 = data[target, 2]
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
            freqrange = np.linspace(
                min(frequencies),
                max(frequencies),
                2 * len(q_data),
            )
            params = data_fit[
                (
                    "fitted_parameters_state_zero"
                    if i == 0
                    else (
                        "fitted_parameters_state_one"
                        if i == 1
                        else "fitted_parameters_state_two"
                    )
                )
            ][target]
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

    if fit is not None:
        for color, best_freq in zip(
            ["green", "orange"][fit.best_freqs01, fit.best_freqs12]
        ):
            fig.add_trace(
                go.Scatter(
                    x=[
                        best_freq[target] * HZ_TO_GHZ,
                        best_freq[target] * HZ_TO_GHZ,
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

            fig.add_vline(
                x=best_freq[target] * HZ_TO_GHZ,
                line=dict(color=color, width=3, dash="dash"),
                row=1,
                col=1,
            )
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "State Zero Frequency [Hz]",
                    "State One Frequency [Hz]",
                    "State Two Frequency [Hz]",
                    "Best 01 readout freq [HZ]",
                    "Best 12 readout freq [HZ]",
                ],
                np.round(
                    [
                        fit_data_0["frequency_state_zero"][target],
                        fit_data_1["frequency_state_one"][target],
                        fit_data_2["frequency_state_two"][target],
                        fit.best_freqs01[target],
                        fit.best_freqs12[target],
                    ]
                ),
            )
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


def _update(results: DispersiveShiftQutritResults, platform: Platform, target: QubitId):
    update.readout_mz1_frequency(results.best_freqs12[target], platform, target)
    update.readout_frequency(results.best_freqs01[target], platform, target)


dispersive_shift_qutrit = Routine(
    _acquisition,
    _fit,
    _plot,
    _update,
)
"""Dispersive shift Routine object."""
