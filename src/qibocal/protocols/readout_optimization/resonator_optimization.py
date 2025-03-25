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
from qibocal.protocols.utils import HZ_TO_GHZ, readout_frequency, table_dict, table_html


@dataclass
class ResonatorOptimizationParameters(Parameters):
    """Resonator optimization runcard inputs"""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    amplitude_start: float
    """Minimum amplitude."""
    amplitude_stop: float
    """Maximum amplitude."""
    amplitude_step: float
    """Step amplitude."""


@dataclass
class ResonatorOptimizationResults(Results):
    """Resonator optimization outputs"""

    fidelities: dict[QubitId, list]
    """Assignment fidelities."""
    best_freq: dict[QubitId, float]
    """Resonator Frequency with the highest assignment fidelity."""
    best_amp: dict[QubitId, list]
    """Amplitude with lowest error."""
    best_angle: dict[QubitId, float]
    """IQ angle that maximes assignment fidelity."""
    best_threshold: dict[QubitId, float]
    """Threshold that maximes assignment fidelity."""


ResonatorOptimizationType = np.dtype(
    [
        ("assignment_fidelity", np.float64),
        ("frequency", np.float64),
        ("amplitude", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype readout optimization."""


@dataclass
class ResonatorOptimizationData(Data):
    """Data class for resonator optimization protocol."""

    resonator_type: str
    """Resonator type."""
    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray[ResonatorOptimizationType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: ResonatorOptimizationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorOptimizationData:
    r"""
    Data acquisition for readout optimization.

    Args:
        params (ResonatorFrequencyParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (list): list of target qubits to perform the action
    """
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    ro_pulses_0 = {}
    ro_pulses_1 = {}
    amplitudes = {}
    freq_sweepers = {}

    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()

    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]

        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse_0 = natives.MZ()[0]
        _, ro_pulse_1 = natives.MZ()[0]

        ro_pulses_0[qubit] = ro_pulse_0
        ro_pulses_1[qubit] = ro_pulse_1
        amplitudes[qubit] = ro_pulse_0.probe.amplitude

        sequence_0.append((ro_channel, ro_pulse_0))

        sequence_1.append((qd_channel, qd_pulse))
        sequence_1.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence_1.append((ro_channel, ro_pulse_1))

        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            values=readout_frequency(qubit, platform) + delta_frequency_range,
            channels=[platform.qubits[qubit].probe],
        )

    amp_sweeper_0 = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.amplitude_start, params.amplitude_stop, params.amplitude_step),
        pulses=[ro_pulses_0[qubit] for qubit in targets],
    )

    amp_sweeper_1 = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.amplitude_start, params.amplitude_stop, params.amplitude_step),
        pulses=[ro_pulses_1[qubit] for qubit in targets],
    )

    data = ResonatorOptimizationData(
        amplitudes=amplitudes,
        resonator_type=platform.resonator_type,
    )

    state0_results = platform.execute(
        [sequence_0],
        [[amp_sweeper_0], [freq_sweepers[q] for q in targets]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    state1_results = platform.execute(
        [sequence_1],
        [[amp_sweeper_1], [freq_sweepers[q] for q in targets]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    # TODO: move analysis in _fit() function
    for qubit in targets:
        ro_pulse_0 = list(sequence_0.channel(platform.qubits[qubit].acquisition))[-1]
        ro_pulse_1 = list(sequence_1.channel(platform.qubits[qubit].acquisition))[-1]

        result0 = np.transpose(state0_results[ro_pulse_0.id], (2, 1, 0, 3))
        result1 = np.transpose(state1_results[ro_pulse_1.id], (2, 1, 0, 3))

        nshots = params.nshots

        for j, freq in enumerate(freq_sweepers[qubit].values):
            for k, amp in enumerate(amp_sweeper_0.values):
                iq_values = np.concatenate((result0[j][k], result1[j][k]))
                states = [0] * nshots + [1] * nshots

                model = QubitFit()
                model.fit(iq_values, np.array(states))
                data.register_qubit(
                    ResonatorOptimizationType,
                    (qubit),
                    dict(
                        assignment_fidelity=np.array([model.assignment_fidelity]),
                        frequency=freq,
                        amplitude=amp,
                        angle=np.array([model.angle]),
                        threshold=np.array([model.threshold]),
                    ),
                )
    return data


def _fit(data: ResonatorOptimizationData) -> ResonatorOptimizationResults:
    qubits = data.qubits
    best_freq = {}
    best_amps = {}
    best_angle = {}
    best_threshold = {}
    highest_fidelity = {}

    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_fid = np.argmax(data_qubit["assignment_fidelity"])
        highest_fidelity[qubit] = data_qubit["assignment_fidelity"][index_best_fid]
        best_freq[qubit] = data_qubit["frequency"][index_best_fid]
        best_amps[qubit] = data_qubit["amplitude"][index_best_fid]
        best_angle[qubit] = data_qubit["angle"][index_best_fid]
        best_threshold[qubit] = data_qubit["threshold"][index_best_fid]

    return ResonatorOptimizationResults(
        best_amp=best_amps,
        fidelities=highest_fidelity,
        best_freq=best_freq,
        best_angle=best_angle,
        best_threshold=best_threshold,
    )


def _plot(
    data: ResonatorOptimizationData, fit: ResonatorOptimizationResults, target: QubitId
):
    """Plotting function for resonator optimization"""

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=1,
    )

    qubit_data = data[target]
    frequencies = qubit_data.frequency
    amplitudes = qubit_data.amplitude
    fidelities = qubit_data.assignment_fidelity

    if fit is not None:
        fig.add_trace(
            go.Heatmap(
                x=amplitudes,
                y=frequencies * HZ_TO_GHZ,
                z=fidelities,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[fit.best_amp],
                y=[fit.best_freq],
                mode="markers",
                marker=dict(
                    size=8,
                    color="black",
                    symbol="cross",
                ),
                name="highest assignment fidelity",
                showlegend=True,
            )
        )

        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Best Readout Amplitude [a.u.]",
                    "Best Readout Frequency [GHz]",
                    "Best Fidelity",
                ],
                [
                    np.round(fit.best_amp[target], 4),
                    np.round(fit.best_freq[target] * HZ_TO_GHZ),
                    fit.fidelities[target],
                ],
            )
        )

        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h"),
        )

        fig.update_xaxes(title_text="Amplitude [a.u.]", row=1, col=1)
        fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=1)

        figures.append(fig)
    return figures, fitting_report


def _update(
    results: ResonatorOptimizationResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.readout_amplitude(results.best_amp[target], platform, target)
    update.readout_frequency(results.best_freq[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


resonator_optimization = Routine(
    _acquisition,
    _fit,
    _plot,
    _update,
)
"""Resonator optimization Routine object"""
