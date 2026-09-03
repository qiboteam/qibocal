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
    RangeLike,
    readout_frequency,
    table_dict,
    table_html,
    to_range,
)

__all__ = ["ro_frequency"]


@dataclass
class ReadoutFrequencyParameters(Parameters):
    """Optimization RO frequency inputs."""

    frequency_range: RangeLike
    """Frequency RangeLike object; for further information, see
    :class:`qibocal.protocols.utils.RangeLike`."""


@dataclass
class ReadoutFrequencyResults(Results):
    """Optimization RO frequency results."""

    highest_fidelities: dict[QubitId, float]
    """Highest Assignment fidelities."""
    best_freq: dict[QubitId, float]
    """Readout Frequency with the highest assignment fidelity."""
    best_angle: dict[QubitId, float]
    """IQ angle that maximes assignment fidelity."""
    best_threshold: dict[QubitId, float]
    """Threshold that maximes assignment fidelity."""
    measured_fidelities: dict[QubitId, list]
    """Measured assignment fidelities."""


# measuring the ground state
ReadoutFrequencyType = np.dtype(
    [
        ("freq", np.float64),
        ("state", np.int8),
        ("i", np.float64),
        ("q", np.float64),
    ]
)
"""Custom dtype for Optimization RO frequency."""


@dataclass
class ReadoutFrequencyData(Data):
    """Optimization RO frequency acquisition outputs."""

    data: dict[QubitId, npt.NDArray[ReadoutFrequencyType]] = field(default_factory=dict)

    def frequencies(self, qubit: QubitId) -> np.ndarray:
        return np.unique(self.data[qubit]["frequency"])

    def select_frequency(self, freq: float) -> "ReadoutFrequencyData":
        selected_data: dict[QubitId, npt.NDArray[ReadoutFrequencyType]] = {
            q: arr[arr["freq"] == freq] for q, arr in self.data.items()
        }
        return ReadoutFrequencyData(data=selected_data)


def _acquisition(
    params: ReadoutFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutFrequencyData:
    r"""
    Data acquisition for readout frequency optimization.
    While sweeping the readout frequency, the routine performs a single shot
    classification and evaluates the assignment fidelity.
    At the end, the readout frequency is updated, choosing the one that has
    the highest assignment fidelity.

    Args:
        params (ReadoutFrequencyParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (list): list of target qubits to perform the action

    """

    data = ReadoutFrequencyData()

    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()

    sweepers: dict[QubitId, Sweeper] = {}
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

        sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            range=to_range(
                spec=params.frequency_range, center=readout_frequency(qubit, platform)
            ),
            channels=[platform.qubits[qubit].probe],
        )

    results = platform.execute(
        [sequence_0, sequence_1],
        [list(sweepers.values())],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    for qubit in targets:
        qubit_freqs = np.asarray(
            [f for f in sweepers[qubit].values for _ in range(params.nshots)] * 2
        )

        total_points = params.nshots * len(sweepers[qubit].values)
        states = np.asarray([[0] * total_points + [1] * total_points]).ravel()

        qubit_ros = probe_pulses_dict[qubit]

        data.register_qubit(
            ReadoutFrequencyType,
            (qubit),
            {
                "freq": qubit_freqs,
                "state": states,
                "i": np.asarray(results[qubit_ros[0].id]).ravel(),
                "q": np.asarray(results[qubit_ros[1].id]).ravel(),
            },
        )

    return data


def _fit(data: ReadoutFrequencyData) -> ReadoutFrequencyResults:
    """Post-Processing for Optimization RO frequency"""

    best_freq: dict[QubitId, float] = {}
    best_angle: dict[QubitId, float] = {}
    best_threshold: dict[QubitId, float] = {}
    highest_ass_fid: dict[QubitId, float] = {}
    ass_fid_dict: dict[QubitId, list] = defaultdict(list)

    for qubit in data.qubits:
        for f in data.frequencies(qubit):
            model = QubitFit()
            model.fit(
                np.stack(
                    (
                        data[qubit]["i"],
                        data[qubit]["q"],
                    ),
                    axis=1,
                ),
                np.asarray(data[qubit]["state"]),
            )
            ass_fid = model.assignment_fidelity
            ass_fid_dict[qubit].append(ass_fid)

            if qubit not in highest_ass_fid or ass_fid > highest_ass_fid[qubit]:
                highest_ass_fid[qubit] = ass_fid
                best_freq[qubit] = f
                best_angle[qubit] = model.angle
                best_threshold[qubit] = model.threshold

    return ReadoutFrequencyResults(
        highest_fidelities=highest_ass_fid,
        best_freq=best_freq,
        best_angle=best_angle,
        best_threshold=best_threshold,
        measured_fidelities=ass_fid_dict,
    )


def _plot(data: ReadoutFrequencyData, fit: ReadoutFrequencyResults, target: QubitId):
    """Plotting function for Optimization RO frequency"""

    figures = []
    opacity = 1
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=1,
    )

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=data.frequencies(target),
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
                target,
                ["Best Readout Frequency [Hz]"],
                [np.round(fit.best_freq[target], 4)],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout Frequencies [GHz]",
        yaxis_title="Assignment Fidelities",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ReadoutFrequencyResults, platform: CalibrationPlatform, target: QubitId
):
    update.readout_frequency(results.best_freq[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)


ro_frequency = Protocol(_acquisition, _fit, _plot, _update)
"""Optimization RO frequency Protocol object"""
