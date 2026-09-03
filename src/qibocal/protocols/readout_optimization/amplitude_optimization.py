from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    Parameter,
    PulseLike,
    PulseSequence,
    Sweeper,
)

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import (
    Range,
    RangeLike,
    table_dict,
    table_html,
    to_range,
)

__all__ = ["ro_amplitude"]


@dataclass
class ReadoutAmplitudeParameters(Parameters):
    """ReadoutAmplitude runcard inputs."""

    amplitude_range: RangeLike
    """Amplitude RangeLike object; for further information, see
    :class:`qibocal.protocols.utils.RangeLike`."""

    @property
    def _amplitude_range(self) -> Range:
        return to_range(self.amplitude_range)


ReadoutAmplitudeType = np.dtype(
    [
        ("amp", np.float64),
        ("state", np.int8),
        ("i", np.float64),
        ("q", np.float64),
    ]
)
"""Custom dtype for Optimization RO amplitude."""


@dataclass
class ReadoutAmplitudeData(Data):
    """Data class for `ro_amplitude` protocol."""

    data: dict[QubitId, npt.NDArray[ReadoutAmplitudeType]] = field(default_factory=dict)

    def amplitudes(self, target: QubitId) -> np.ndarray:
        return np.unique(self.data[target]["amp"])

    def select_amplitude(self, ampl: float) -> "ReadoutAmplitudeData":
        selected_data: dict[QubitId, npt.NDArray[ReadoutAmplitudeType]] = {
            q: arr[arr["amp"] == ampl] for q, arr in self.data.items()
        }
        return ReadoutAmplitudeData(data=selected_data)


@dataclass
class ReadoutAmplitudeResults(Results):
    """Result class for `ro_amplitude` protocol."""

    highest_assignment_fidelities: dict[QubitId, float]
    """Highest assignment fidelities"""
    best_amp: dict[QubitId, float]
    """Amplitude with lowest error"""
    best_angle: dict[QubitId, float]
    """IQ angle that gives lower error."""
    best_threshold: dict[QubitId, float]
    """Thershold that gives lower error."""
    measured_ass_fidelities: dict[QubitId, list]
    """Measured assignment fidelities"""


def _acquisition(
    params: ReadoutAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutAmplitudeData:
    r"""
    Data acquisition for resoantor amplitude optmization.
    This protocol sweeps the readout amplitude performing a classification routine
    and evaluating the error probability at each step. The sweep will be interrupted
    if the probability error is less than the `error_threshold`.

    Args:
        params (:class:`ReadoutAmplitudeParameters`): input parameters
        platform (:class:`CalibrationPlatform`): Qibolab's platform
        targets (list): list of QubitIds to be characterized

    Returns:
        data (:class:`ReadoutAmplitudeData`)
    """

    data = ReadoutAmplitudeData()

    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()

    probe_pulses_dict: dict[QubitId, dict[int, PulseLike]] = {}
    for qubit in targets:
        # Get the native gates and readout channel/pulse for the current qubit
        natives = platform.natives.single_qubit[qubit]
        ro_channel, ro_pulse_0 = natives.MZ()[0]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_pulse_1 = ro_pulse_0.new()

        # appending the probe pulses to the dictionary for later sweeping
        probe_pulses_dict[qubit] = {
            0: ro_pulse_0,
            1: ro_pulse_1,
        }

        # measuring the ground state
        sequence_0.append((ro_channel, ro_pulse_0))

        # preparaing and measuring the excited state
        sequence_1 += PulseSequence([(qd_channel, qd_pulse)]) | PulseSequence(
            [(ro_channel, ro_pulse_1)]
        )

    amplitude_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=params._amplitude_range,
        pulses=[
            pulse
            for qubit_pulses in probe_pulses_dict.values()
            for pulse in qubit_pulses.values()
        ],
    )

    results = platform.execute(
        [sequence_0, sequence_1],
        [[amplitude_sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    amplitudes = np.asarray(
        [a for a in amplitude_sweeper.values for _ in range(params.nshots)] * 2
    )

    total_points = params.nshots * len(amplitude_sweeper.values)
    states = np.asarray([[0] * total_points + [1] * total_points]).ravel()

    # saving measurement results for each qubit
    for qubit in targets:
        qubit_ros = probe_pulses_dict[qubit]

        data.register_qubit(
            ReadoutAmplitudeType,
            (qubit),
            {
                "amp": amplitudes,
                "state": states,
                "i": np.asarray(results[qubit_ros[0].id]).ravel(),
                "q": np.asarray(results[qubit_ros[1].id]).ravel(),
            },
        )

    return data


def _fit(data: ReadoutAmplitudeData) -> ReadoutAmplitudeResults:
    """Post-Processing for Optimization RO amplitude"""

    qubits = data.qubits
    best_amps: dict[QubitId, float] = {}
    best_angle: dict[QubitId, float] = {}
    best_threshold: dict[QubitId, float] = {}
    highest_ass_fid: dict[QubitId, float] = {}
    ass_fid_dict: dict[QubitId, list] = defaultdict(list)

    for a in data.amplitudes(qubits[0]):
        selected_data = data.select_amplitude(a)

        for qb in qubits:
            model = QubitFit()
            model.fit(
                np.stack(
                    (
                        selected_data[qb]["i"],
                        selected_data[qb]["q"],
                    ),
                    axis=1,
                ),
                np.asarray(selected_data[qb]["state"]),
            )
            ass_fid = model.assignment_fidelity
            ass_fid_dict[qb].append(ass_fid)

            if qb not in highest_ass_fid or ass_fid > highest_ass_fid[qb]:
                highest_ass_fid[qb] = ass_fid
                best_amps[qb] = a
                best_angle[qb] = model.angle
                best_threshold[qb] = model.threshold

    return ReadoutAmplitudeResults(
        highest_assignment_fidelities=highest_ass_fid,
        best_amp=best_amps,
        best_angle=best_angle,
        best_threshold=best_threshold,
        measured_ass_fidelities=ass_fid_dict,
    )


def _plot(data: ReadoutAmplitudeData, fit: ReadoutAmplitudeResults, target: QubitId):
    """Plotting function for Optimization RO amplitude."""
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
                x=data.amplitudes(target),
                y=fit.measured_ass_fidelities[target],
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
                ["Best Readout Amplitude [a.u.]"],
                [np.round(fit.best_amp[target], 4)],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout Amplitude [a.u.]",
        yaxis_title="Assignment Fidelities",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ReadoutAmplitudeResults, platform: CalibrationPlatform, target: QubitId
):
    update.readout_amplitude(results.best_amp[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


ro_amplitude = Protocol(_acquisition, _fit, _plot, _update)
"""Readout Amplitude Protocol  object."""
