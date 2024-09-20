from dataclasses import asdict, dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.identifier import QubitId
from qibolab.platform import Platform
from qibolab.sequence import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

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

    frequency_state_zero: dict[QubitId, float]
    """State zero frequency."""
    frequency_state_one: dict[QubitId, float]
    """State one frequency."""
    fitted_parameters_state_zero: dict[QubitId, list[float]]
    """Fitted parameters state zero."""
    fitted_parameters_state_one: dict[QubitId, list[float]]
    """Fitted parameters state one."""
    best_freq: dict[QubitId, float]
    """Readout frequency that maximizes the distance of ground and excited states in iq-plane"""

    @property
    def state_zero(self):
        return {key: value for key, value in asdict(self).items() if "zero" in key}

    @property
    def state_one(self):
        return {key: value for key, value in asdict(self).items() if "one" in key}


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

    frequency_0 = {}
    frequency_1 = {}
    best_freqs = {}
    fitted_parameters_0 = {}
    fitted_parameters_1 = {}

    for i in range(2):
        for qubit in qubits:
            data_i = data[qubit, i]
            fit_result = lorentzian_fit(
                data_i, resonator_type=data.resonator_type, fit="resonator"
            )
            if fit_result is not None:
                if i == 0:
                    frequency_0[qubit], fitted_parameters_0[qubit], _ = fit_result
                else:
                    frequency_1[qubit], fitted_parameters_1[qubit], _ = fit_result

            i_measures = data_i.i
            q_measures = data_i.q

            iq_couples[i].append(np.stack((i_measures, q_measures), axis=-1))
        # for each qubit find the iq couple of 0-1 states that maximize the distance
    iq_couples = np.array(iq_couples)

    for idx, qubit in enumerate(qubits):
        frequencies = data[qubit, 0].freq

        max_index = np.argmax(
            np.linalg.norm(iq_couples[0][idx] - iq_couples[1][idx], axis=-1)
        )
        best_freqs[qubit] = frequencies[max_index]

    return DispersiveShiftResults(
        frequency_state_zero=frequency_0,
        frequency_state_one=frequency_1,
        fitted_parameters_state_one=fitted_parameters_1,
        fitted_parameters_state_zero=fitted_parameters_0,
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
    # iterate over multiple data folders

    fitting_report = ""

    data_0 = data[target, 0]
    data_1 = data[target, 1]
    fit_data_0 = fit.state_zero if fit is not None else None
    fit_data_1 = fit.state_one if fit is not None else None

    for i, label, q_data, data_fit in list(
        zip(
            (0, 1),
            ("State 0", "State 1"),
            (data_0, data_1),
            (fit_data_0, fit_data_1),
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
                    else "fitted_parameters_state_one"
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

        fig.add_vline(
            x=fit.best_freq[target] * HZ_TO_GHZ,
            line=dict(color="orange", width=3, dash="dash"),
            row=1,
            col=1,
        )
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "State Zero Frequency [Hz]",
                    "State One Frequency [Hz]",
                    "Chi [Hz]",
                    "Best Frequency [Hz]",
                ],
                np.round(
                    [
                        fit_data_0["frequency_state_zero"][target],
                        fit_data_1["frequency_state_one"][target],
                        (
                            fit_data_0["frequency_state_zero"][target]
                            - fit_data_1["frequency_state_one"][target]
                        )
                        / 2,
                        fit.best_freq[target],
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


def _update(results: DispersiveShiftResults, platform: Platform, target: QubitId):
    update.readout_frequency(results.best_freq[target], platform, target)


dispersive_shift = Routine(_acquisition, _fit, _plot, _update)
"""Dispersive shift Routine object."""
