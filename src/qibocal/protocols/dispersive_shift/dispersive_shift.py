from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.constants
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    lorentzian_fit,
    lorentzian_with_linear_background,
    readout_frequency,
    table_dict,
    table_html,
)
from qibocal.result import magnitude, phase, unpack

__all__ = ["dispersive_shift", "DispersiveShiftData", "DispersiveShiftParameters"]


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
    params: DispersiveShiftParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> DispersiveShiftData:
    r"""
    Data acquisition for dispersive shift experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (DispersiveShiftParameters): experiment's parameters
        platform (CalibrationPlatform): Qibolab platform object
        targets (list): list of target qubits to perform the action
    """

    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        sequence_0 += natives.MZ()
        sequence_1 += natives.RX() | natives.MZ()

    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    data = DispersiveShiftData(resonator_type=platform.resonator_type)

    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=readout_frequency(q, platform) + delta_frequency_range,
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]

    results = platform.execute(
        [sequence_0, sequence_1],
        [sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for qubit in targets:
        for state, sequence in enumerate([sequence_0, sequence_1]):
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            result = results[ro_pulse.id]
            i, q = unpack(result)
            data.register_qubit(
                DispersiveShiftType,
                (qubit, state),
                dict(
                    freq=readout_frequency(qubit, platform) + delta_frequency_range,
                    signal=magnitude(result),
                    phase=phase(result),
                    i=i,
                    q=q,
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
    color_map = {
        0: "blue",
        1: "red",
    }
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

    for state in (0, 1):
        label = f"State {state}"
        data_group = f"state{state}-data"
        fit_group = f"state{state}-fit"
        q_data = data[target, state]

        frequencies = q_data.freq * scipy.constants.nano
        for col, y in enumerate([q_data.signal, q_data.phase], start=1):
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=y,
                    name=f"{label} data",
                    showlegend=(col == 1),
                    legendgroup=data_group,
                    mode="markers",
                    marker=dict(
                        color=color_map[state],
                        size=5,
                        symbol="circle",
                    ),
                ),
                row=1,
                col=col,
            )
        if fit is not None and fit.frequencies[target] is not None:
            freqrange = np.linspace(
                min(frequencies),
                max(frequencies),
                2 * len(q_data),
            )
            params = fit.fitted_parameters[target][state]
            fig.add_trace(
                go.Scatter(
                    x=freqrange,
                    y=lorentzian_with_linear_background(freqrange, *params),
                    name=f"{label} fit",
                    showlegend=True,
                    legendgroup=fit_group,
                    line=go.scatter.Line(color=color_map[state]),
                ),
                row=1,
                col=1,
            )

    fitting_report = ""
    if fit is not None:
        bestfit_group = "bestfit"
        all_signals = np.concatenate((data[target, 0].signal, data[target, 1].signal))
        fig.add_trace(
            go.Scatter(
                x=[fit.best_freq[target] * scipy.constants.nano] * 2,
                y=[all_signals.min(), all_signals.max()],
                mode="lines",
                line=go.scatter.Line(color="black", width=3, dash="dash"),
                name="Best frequency",
                legendgroup=bestfit_group,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        all_phases = np.concatenate((data[target, 0].phase, data[target, 1].phase))
        fig.add_trace(
            go.Scatter(
                x=[fit.best_freq[target] * scipy.constants.nano] * 2,
                y=[all_phases.min(), all_phases.max()],
                mode="lines",
                line=go.scatter.Line(color="black", width=3, dash="dash"),
                legendgroup=bestfit_group,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        if fit.frequencies[target] is not None:
            table_entries = [
                "Ground state Frequency [Hz]",
                "Excited state Frequency [Hz]",
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
        else:
            table_entries = ["Best Frequency [Hz]"]
            table_values = np.round([fit.best_freq[target]])

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


def _update(
    results: DispersiveShiftResults, platform: CalibrationPlatform, target: QubitId
):
    update.readout_frequency(results.best_freq[target], platform, target)
    if results.frequencies[target] is not None:
        delta = (
            platform.calibration.single_qubits[target].qubit.frequency_01
            - results.frequencies[target][0]
        )
        g = np.sqrt(np.abs(results.chi(target) * delta))
        update.readout_coupling(g, platform, target)
        update.dressed_resonator_frequency(
            results.frequencies[target][0], platform, target
        )
        platform.calibration.single_qubits[target].readout.qudits_frequency[1] = (
            results.frequencies[target][1]
        )


dispersive_shift = Protocol(_acquisition, _fit, _plot, _update)
"""Dispersive shift Protocol object."""
