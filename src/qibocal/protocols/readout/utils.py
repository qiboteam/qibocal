from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    PulseId,
    PulseLike,
    PulseSequence,
    Result,
)

from qibocal.auto.operation import Data, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import table_dict, table_html


@dataclass
class ReadoutResults(Results):
    """Optimization RO frequency results."""

    best_swept_param: dict[QubitId, float]
    """Best swept parameter value."""
    highest_fidelities: dict[QubitId, float]
    """Highest Assignment fidelities."""
    best_angle: dict[QubitId, float]
    """IQ angle that maximes assignment fidelity."""
    best_threshold: dict[QubitId, float]
    """Threshold that maximes assignment fidelity."""
    measured_fidelities: dict[QubitId, list]
    """Measured assignment fidelities."""


@dataclass
class ReadoutData(Data):
    """Optimization RO frequency acquisition outputs."""

    swept_parameter: dict[QubitId, list[float]] = field(default_factory=dict)
    """List of parameter swept for each qubit."""
    data: dict[tuple[QubitId, int, float], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    """Measured data for each qubit, with shape (Nshots, N_freq_sweep, 2)."""
    classification_info: dict[QubitId, list[list[float]]] = field(default_factory=dict)
    """Classification information for each qubit (Assignment Fidelity, Angle and
    Threshold of the classifier)."""
    save_iq: bool = False
    """Whether to save the IQ data during the acquisition."""


def readout_sequence(
    platform: CalibrationPlatform, targets: list[QubitId]
) -> tuple[list[PulseSequence], dict[QubitId, dict[int, PulseLike]]]:
    """Build readout sequences for ground- and excited-state measurements."""

    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()

    probe_pulses_dict: dict[QubitId, dict[int, PulseLike]] = {}
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse_0 = natives.MZ()[0]
        ro_pulse_1 = ro_pulse_0.new()

        # measuring the ground state
        sequence_0.append((ro_channel, ro_pulse_0))

        # preparing and measuring the excited state
        sequence_1 += PulseSequence([(qd_channel, qd_pulse)]) | PulseSequence(
            [(ro_channel, ro_pulse_1)]
        )

        probe_pulses_dict[qubit] = {
            0: ro_pulse_0,
            1: ro_pulse_1,
        }

    return [sequence_0, sequence_1], probe_pulses_dict


def fit_classification_model(
    measured_0: npt.NDArray[np.float64], measured_1: npt.NDArray[np.float64]
) -> QubitFit:
    """Fit a binary readout classification model to IQ samples.

    It returns the fitted qubit classification model.
    """

    model = QubitFit()
    model.fit(
        np.concatenate((measured_0, measured_1)),
        np.asarray([0] * len(measured_0) + [1] * len(measured_1)),
    )
    return model


def save_data(
    targets: list[QubitId],
    parameter_dict: dict[QubitId, list[float]],
    pulses_dict: dict[QubitId, dict[int, PulseLike]],
    results: dict[PulseId, Result],
    save_iq: bool,
) -> tuple[
    dict[tuple[QubitId, int, float], npt.NDArray[np.float64]],
    dict[QubitId, list[list[float]]],
]:
    """Extract and optionally classify readout data for each parameter value.

    It returns a mapping keyed by ``(qubit, state, parameter)`` containing either raw
    IQ samples or classification metrics.
    """

    data: dict[tuple[QubitId, int, float], npt.NDArray[np.float64]] = {}
    fit_res: dict[QubitId, list[list[float]]] = {}
    # saving measurement results for each qubit
    for qubit in targets:
        qubit_ros = pulses_dict[qubit]

        state_0 = np.asarray(results[qubit_ros[0].id])
        state_1 = np.asarray(results[qubit_ros[1].id])
        for idx, param in enumerate(parameter_dict[qubit]):
            sweep_state_0 = state_0[:, idx, :]
            sweep_state_1 = state_1[:, idx, :]

            if save_iq:
                data[qubit, 0, param] = sweep_state_0
                data[qubit, 1, param] = sweep_state_1

            fitted_result = fit_classification_model(sweep_state_0, sweep_state_1)

            fit_res.setdefault(qubit, []).append(
                [
                    fitted_result.assignment_fidelity,
                    fitted_result.angle,
                    fitted_result.threshold,
                ]
            )

    return data, fit_res


def readout_fit(data: ReadoutData) -> ReadoutResults:
    """Fit the readout data for each qubit and return the results."""

    best_param: dict[QubitId, float] = {}
    best_angle: dict[QubitId, float] = {}
    best_threshold: dict[QubitId, float] = {}
    highest_ass_fid: dict[QubitId, float] = {}
    ass_fid_dict: dict[QubitId, list[float]] = {}

    for qb in data.qubits:
        fit_res_array = np.asarray(data.classification_info[qb])
        ass_fid_dict[qb] = fit_res_array[:, 0].tolist()

        # maximize assignment fidelity to find the best parameter, angle and threshold
        max_fidelity_idx = np.argmax(fit_res_array[:, 0])

        highest_ass_fid[qb] = fit_res_array[max_fidelity_idx, 0]
        best_param[qb] = data.swept_parameter[qb][max_fidelity_idx]
        best_angle[qb] = fit_res_array[max_fidelity_idx, 1]
        best_threshold[qb] = fit_res_array[max_fidelity_idx, 2]

    return ReadoutResults(
        best_swept_param=best_param,
        highest_fidelities=highest_ass_fid,
        best_angle=best_angle,
        best_threshold=best_threshold,
        measured_fidelities=ass_fid_dict,
    )


def readout_plot(
    data: ReadoutData,
    fit: ReadoutResults,
    target: QubitId,
    label: str,
) -> tuple[list[go.Figure], str | None]:
    """Create an assignment-fidelity plot for a target qubit."""

    figures = []
    opacity = 1
    fitting_report = None
    fig = make_subplots(
        rows=1,
        cols=1,
    )
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=data.swept_parameter[target],
                y=fit.measured_fidelities[target],
                opacity=opacity,
                showlegend=True,
                name="Assignment Fidelities",
                mode="markers+lines",
            ),
            row=1,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                [target],
                ["Best Readout " + label],
                [np.round(fit.best_swept_param[target], 4)],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout " + label,
        yaxis_title="Assignment Fidelities",
    )
    figures.append(fig)

    return figures, fitting_report
