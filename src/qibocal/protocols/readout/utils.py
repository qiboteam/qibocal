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
    """List of parameter values swept for each qubit."""
    data: dict[tuple[QubitId, int, float], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    """Measured data for each qubit, with shape (Nshots, N_freq_sweep, 2)."""
    classification_info: dict[QubitId, list[list[float]]] = field(default_factory=dict)
    """Classification information for each qubit (Assignment Fidelity, Angle and
    Threshold of the classifier)."""
    save_iq: bool = False
    """Whether to save the IQ data during the acquisition."""

    # I need to overwrite this property since data might be empty if
    # saving flag `save_iq` is set to False.
    # In that case, the `data` attribute will be empty and the `qubits`
    # property will return an empty list.
    @property
    def qubits(self) -> list[QubitId]:
        """Return the list of qubits for which data was acquired."""
        return list(self.swept_parameter.keys())


def base_sequence(
    platform: CalibrationPlatform, targets: list[QubitId]
) -> tuple[list[PulseSequence], dict[QubitId, dict[int, PulseLike]]]:
    """Build readout sequences for ground- and excited-state measurements."""

    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()

    readouts: dict[QubitId, dict[int, PulseLike]] = {}
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        # measuring the ground state
        sequence_0 += natives.MZ()

        # preparing and measuring the excited state
        sequence_1 += natives.RX() | natives.MZ()

        readouts[qubit] = {
            0: sequence_0[-1][1],
            1: sequence_1[-1][1],
        }

    return [sequence_0, sequence_1], readouts


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


def fit_readout_classification_models(
    targets: list[QubitId],
    parameter_dict: dict[QubitId, list[float]],
    pulses_dict: dict[QubitId, dict[int, PulseLike]],
    results: dict[PulseId, Result],
    save_iq: bool,
) -> ReadoutData:
    """Fit readout classification models for each target qubit and parameter sweep.

    For each qubit and swept readout parameter value, the function loads the
    ground- and excited-state IQ data from the acquisition results, optionally
    stores the raw IQ samples, and fits a binary classifier. The fitted
    assignment fidelity, IQ rotation angle, and discrimination threshold are
    collected for each sweep point and returned together with the optional IQ
    samples.
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

    return ReadoutData(
        swept_parameter=parameter_dict,
        data=data,
        classification_info=fit_res,
        save_iq=save_iq,
    )


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
