from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import readout_frequency, table_dict, table_html

__all__ = ["resonator_frequency"]


@dataclass
class ResonatorFrequencyParameters(Parameters):
    """Optimization RO frequency inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""


@dataclass
class ResonatorFrequencyResults(Results):
    """Optimization RO frequency results."""

    fidelities: dict[QubitId, list]
    """Assignment fidelities."""
    best_freq: dict[QubitId, float]
    """Resonator Frequency with the highest assignment fidelity."""
    best_angle: dict[QubitId, float]
    """IQ angle that maximes assignment fidelity."""
    best_threshold: dict[QubitId, float]
    """Threshold that maximes assignment fidelity."""


ResonatorFrequencyType = np.dtype(
    [
        ("frequency", np.float64),
        ("assignment_fidelity", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype for Optimization RO frequency."""


@dataclass
class ResonatorFrequencyData(Data):
    """Optimization RO frequency acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[QubitId, npt.NDArray[ResonatorFrequencyType]] = field(
        default_factory=dict
    )

    def unique_freqs(self, qubit: QubitId) -> np.ndarray:
        return np.unique(self.data[qubit]["frequency"])


def _acquisition(
    params: ResonatorFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorFrequencyData:
    r"""
    Data acquisition for readout frequency optimization.
    While sweeping the readout frequency, the routine performs a single shot
    classification and evaluates the assignment fidelity.
    At the end, the readout frequency is updated, choosing the one that has
    the highest assignment fidelity.

    Args:
        params (ResonatorFrequencyParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (list): list of target qubits to perform the action

    """

    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()

    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse_0 = natives.MZ()[0]
        _, ro_pulse_1 = natives.MZ()[0]

        sequence_0.append((ro_channel, ro_pulse_0))

        sequence_1.append((qd_channel, qd_pulse))
        sequence_1.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence_1.append((ro_channel, ro_pulse_1))

    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    data = ResonatorFrequencyData(resonator_type=platform.resonator_type)

    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=readout_frequency(q, platform) + delta_frequency_range,
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]

    state0_results = platform.execute(
        [sequence_0],
        [sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    state1_results = platform.execute(
        [sequence_1],
        [sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    # TODO: move QubitFit() and anlysis in _fit()
    nshots = params.nshots
    for q, qubit in enumerate(targets):
        result0 = np.transpose(state0_results[ro_pulse_0.id], (1, 0, 2))
        result1 = np.transpose(state1_results[ro_pulse_1.id], (1, 0, 2))

        for j, freq in enumerate(sweepers[q].values):
            iq_values = np.concatenate([result0[j], result1[j]], axis=0)
            states = [0] * nshots + [1] * nshots

            model = QubitFit()
            model.fit(iq_values, np.array(states))

            data.register_qubit(
                ResonatorFrequencyType,
                (qubit),
                dict(
                    frequency=np.array([freq]),
                    assignment_fidelity=np.array([model.assignment_fidelity]),
                    angle=np.array([model.angle]),
                    threshold=np.array([model.threshold]),
                ),
            )

    return data


def _fit(data: ResonatorFrequencyData) -> ResonatorFrequencyResults:
    """Post-Processing for Optimization RO frequency"""

    # TODO: change data.qubits
    qubits = data.qubits
    best_freq = {}
    best_angle = {}
    best_threshold = {}
    highest_fidelity = {}

    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_fid = np.argmax(data_qubit["assignment_fidelity"])
        highest_fidelity[qubit] = data_qubit["assignment_fidelity"][index_best_fid]
        best_freq[qubit] = data_qubit["frequency"][index_best_fid]
        best_angle[qubit] = data_qubit["angle"][index_best_fid]
        best_threshold[qubit] = data_qubit["threshold"][index_best_fid]

    return ResonatorFrequencyResults(
        fidelities=highest_fidelity,
        best_freq=best_freq,
        best_angle=best_angle,
        best_threshold=best_threshold,
    )


def _plot(
    data: ResonatorFrequencyData, fit: ResonatorFrequencyResults, target: QubitId
):
    """Plotting function for Optimization RO frequency"""

    figures = []
    freqs = data[target]["frequency"]
    opacity = 1
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=1,
    )

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=data[target]["assignment_fidelity"],
                opacity=opacity,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                target,
                "Best Resonator Frequency [Hz]",
                np.round(fit.best_freq[target], 4),
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Resonator Frequencies [GHz]",
        yaxis_title="Assignment Fidelities",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ResonatorFrequencyResults, platform: CalibrationPlatform, target: QubitId
):
    update.readout_frequency(results.best_freq[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)


resonator_frequency = Routine(_acquisition, _fit, _plot, _update)
"""Optimization RO frequency Routine object"""
