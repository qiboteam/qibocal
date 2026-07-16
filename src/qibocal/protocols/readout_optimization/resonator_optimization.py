from dataclasses import dataclass, field
from itertools import product

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from ... import update
from ...auto.operation import Data, Parameters, Protocol, QubitId, QubitPairId, Results
from ...calibration import CalibrationPlatform
from ...config import log
from ...fitting.classifier.qubit_fit import QubitFit
from ..utils import (
    HZ_TO_GHZ,
    classify,
    compute_assignment_fidelity,
    compute_qnd,
    readout_frequency,
    table_dict,
    table_html,
)

__all__ = ["resonator_optimization"]


@dataclass
class ResonatorOptimizationParameters(Parameters):
    """Resonator optimization runcard inputs"""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    amplitude_min: float
    """Minimum amplitude."""
    amplitude_max: float
    """Maximum amplitude."""
    amplitude_step: float
    """Step amplitude."""
    delay: float = 0
    """Delay between readouts, could account for resonator depletion or not [ns]."""

    @property
    def frequency_span(self) -> np.ndarray:
        return np.arange(-self.freq_width / 2, self.freq_width / 2, self.freq_step)


@dataclass
class ResonatorOptimizationResults(Results):
    """Resonator optimization outputs"""

    data: dict[tuple[QubitId, str], np.ndarray]
    """Dict storing fidelity, qnd and qnd-pi."""
    fidelity: dict[QubitId, float]
    """Assignment fidelity at optimal readout point."""
    qnd: dict[QubitId, float]
    """QND at optimal readout point."""
    qnd_pi: dict[QubitId, float]
    """QND-pi at optimal readout point."""
    frequency: dict[QubitId, float]
    """Frequency at optimal readout point."""
    amplitude: dict[QubitId, float]
    """Amplitude at optimal readout point."""
    angle: dict[QubitId, float]
    """Angle at optimal readout point."""
    threshold: dict[QubitId, float]
    """Threshold at optimal readout point."""

    def __contains__(self, key: QubitId | QubitPairId | tuple[QubitId, ...]) -> bool:
        """Check whether plotting data is available for qubit."""
        return all(
            (key, metric) in self.data for metric in ["fidelity", "qnd", "qnd-pi"]
        )


@dataclass
class ResonatorOptimizationData(Data):
    """Data class for resonator optimization protocol."""

    frequencies_swept: dict[QubitId, list[float]] = field(default_factory=dict)
    """Frequency swept for each qubit."""
    amplitudes_swept: list[float] = field(default_factory=list)
    """Amplitude swept (same for all qubits)."""
    data: dict[tuple, np.ndarray] = field(default_factory=dict)
    """Raw data acquired"""

    def grid(self, qubit: QubitId) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.meshgrid(self.frequencies_swept[qubit], self.amplitudes_swept)
        return x.ravel(), y.ravel()


def _acquisition(
    params: ResonatorOptimizationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorOptimizationData:
    """Protocol to optimize readout frequency and readout amplitude.

    After preparing either state 0 or state 1 we perform two consecutive measurements to
    evaluate QND. Additionally we apply a pi pulse and we perform a third measurement to
    evaluate the QND-pi following https://arxiv.org/pdf/2110.04285"""

    ro_pulses = {}
    sequences = []
    for state in [0, 1]:
        sequence = PulseSequence()
        for qubit in targets:
            natives = platform.natives.single_qubit[qubit]
            ro_channel = platform.qubits[qubit].acquisition
            drive_channel = platform.qubits[qubit].drive
            mz_pulses = [natives.MZ()[0][1] for _ in range(3)]
            for m, pulse in enumerate(mz_pulses):
                ro_pulses[qubit, state, m] = pulse
            ro_pulse_m1, ro_pulse_m2, ro_pulse_m3 = mz_pulses
            rx_duration = natives.RX().duration

            if state == 1:
                sequence += natives.RX()
                sequence.append((ro_channel, Delay(duration=rx_duration)))
            sequence.append((ro_channel, ro_pulse_m1))
            sequence.append((ro_channel, Delay(duration=params.delay)))
            sequence.append((ro_channel, ro_pulse_m2))
            sequence.append(
                (
                    drive_channel,
                    Delay(
                        duration=params.delay
                        + ro_pulse_m1.duration
                        + ro_pulse_m2.duration
                    ),
                )
            )
            sequence += natives.RX()
            sequence.append((ro_channel, Delay(duration=rx_duration + params.delay)))
            sequence.append((ro_channel, ro_pulse_m3))
        sequences.append(sequence)

    data = ResonatorOptimizationData()

    freq_sweepers = {}
    for qubit in targets:
        freqs = readout_frequency(qubit, platform) + params.frequency_span
        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            values=freqs,
            channels=[platform.qubits[qubit].probe],
        )
        data.frequencies_swept[qubit] = freqs.tolist()

    amp_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(
            params.amplitude_min,
            params.amplitude_max,
            params.amplitude_step,
        ),
        pulses=list(ro_pulses.values()),
    )
    data.amplitudes_swept = amp_sweeper.values.tolist()

    results = platform.execute(
        sequences,
        [[amp_sweeper], [freq_sweepers[qubit] for qubit in targets]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for state in [0, 1]:
        for target in targets:
            for m in range(3):
                data.data[target, state, m] = results[ro_pulses[target, state, m].id]

    return data


def _fit(data: ResonatorOptimizationData) -> ResonatorOptimizationResults:
    qubits = data.qubits
    arr = {}
    frequency = {}
    amplitude = {}
    angle = {}
    threshold = {}
    best_fidelity = {}
    best_qnd = {}
    best_qnd_pi = {}

    for qubit in qubits:
        freq_vals = data.frequencies_swept[qubit]
        amp_vals = data.amplitudes_swept
        shape = (len(amp_vals), len(freq_vals))
        grid_keys = ["fidelity", "angle", "threshold", "qnd", "qnd-pi"]
        grids = {key: np.zeros(shape) for key in grid_keys}
        for j, k in product(range(len(amp_vals)), range(len(freq_vals))):
            measurements = {
                (m, state): data.data[qubit, state, m][:, j, k, :]
                for state in (0, 1)
                for m in range(3)
            }

            iq_values = np.concatenate((measurements[0, 0], measurements[0, 1]))
            nshots = iq_values.shape[0] // 2
            states = [0] * nshots + [1] * nshots

            model = QubitFit()
            model.fit(iq_values, np.array(states))
            grids["angle"][j, k] = model.angle
            grids["threshold"][j, k] = model.threshold

            classified_states = {
                key: classify(val, model.angle, model.threshold)
                for key, val in measurements.items()
            }

            grids["fidelity"][j, k] = compute_assignment_fidelity(
                classified_states[0, 1], classified_states[0, 0]
            )
            grids["qnd"][j, k], _, _ = compute_qnd(
                classified_states[0, 1],
                classified_states[0, 0],
                classified_states[1, 1],
                classified_states[1, 0],
            )
            # for m3 we swap them because we apply a pi pulse
            grids["qnd-pi"][j, k], _, _ = compute_qnd(
                classified_states[1, 1],
                classified_states[1, 0],
                classified_states[2, 0],
                classified_states[2, 1],
                pi=True,
            )
        arr[qubit, "fidelity"] = grids["fidelity"]
        arr[qubit, "qnd"] = grids["qnd"]
        arr[qubit, "qnd-pi"] = grids["qnd-pi"]

        averaged_qnd = (arr[qubit, "qnd"] + arr[qubit, "qnd-pi"]) / 2

        # mask values where fidelity is below 80%
        averaged_qnd[grids["fidelity"] < 0.8] = np.nan
        # exclude values where QND is larger than 1
        averaged_qnd[averaged_qnd > 1] = np.nan
        try:
            i, j = np.unravel_index(np.nanargmax(averaged_qnd), averaged_qnd.shape)
            best_fidelity[qubit] = grids["fidelity"][i, j]
            best_qnd[qubit] = grids["qnd"][i, j]
            best_qnd_pi[qubit] = grids["qnd-pi"][i, j]
            frequency[qubit] = freq_vals[j]
            amplitude[qubit] = amp_vals[i]
            angle[qubit] = grids["angle"][i, j]
            threshold[qubit] = grids["threshold"][i, j]
        except ValueError:
            log.warning("Fitting error.")

    return ResonatorOptimizationResults(
        data=arr,
        fidelity=best_fidelity,
        qnd=best_qnd,
        qnd_pi=best_qnd_pi,
        frequency=frequency,
        amplitude=amplitude,
        angle=angle,
        threshold=threshold,
    )


def _plot(
    data: ResonatorOptimizationData, fit: ResonatorOptimizationResults, target: QubitId
):
    """Plotting function for resonator optimization"""
    figures = []
    fitting_report = ""
    ncols = 3
    fig = make_subplots(
        rows=1,
        cols=ncols,
        subplot_titles=("Fidelity", "QND", "QND Pi"),
    )

    # use fit.frequency as proxy for having found a best point
    has_best_point = fit is not None and target in fit.frequency

    if fit is not None:
        fig.add_trace(
            go.Heatmap(
                x=np.array(data.frequencies_swept[target]) * HZ_TO_GHZ,
                y=data.amplitudes_swept,
                z=fit.data[target, "fidelity"],
                coloraxis="coloraxis",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=np.array(data.frequencies_swept[target]) * HZ_TO_GHZ,
                y=data.amplitudes_swept,
                z=fit.data[target, "qnd"],
                coloraxis="coloraxis",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Heatmap(
                x=np.array(data.frequencies_swept[target]) * HZ_TO_GHZ,
                y=data.amplitudes_swept,
                z=fit.data[target, "qnd-pi"],
                coloraxis="coloraxis",
            ),
            row=1,
            col=3,
        )

        # Layout updates
        fig.update_layout(
            yaxis_title="Amplitude [a.u.]",
            xaxis_title="Frequency [GHz]",
            xaxis2_title="Frequency [GHz]",
            xaxis3_title="Frequency [GHz]",
            coloraxis=dict(colorscale="Viridis", cmin=0, cmax=1),
            legend=dict(orientation="h"),
        )

        if has_best_point:
            for col in range(1, ncols + 1):
                fig.add_trace(
                    go.Scatter(
                        x=[fit.frequency[target] * HZ_TO_GHZ],
                        y=[fit.amplitude[target]],
                        mode="markers",
                        marker=dict(size=8, color="black", symbol="cross"),
                        name="Best Readout Point",
                        showlegend=True if col == 1 else False,
                    ),
                    row=1,
                    col=col,
                )

            fitting_report = table_html(
                table_dict(
                    target,
                    [
                        "Assignment-Fidelity",
                        "QND",
                        "QND Pi",
                        "Best Frequency [Hz]",
                        "Best Amplitude",
                    ],
                    [
                        np.round(fit.fidelity[target], 4),
                        np.round(fit.qnd[target], 4),
                        np.round(fit.qnd_pi[target], 4),
                        np.round(fit.frequency[target], 4),
                        np.round(fit.amplitude[target], 4),
                    ],
                )
            )
        else:
            fitting_report = "An error occurred when performing the fit."

        figures.append(fig)
    return figures, fitting_report


def _update(
    results: ResonatorOptimizationResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.readout_amplitude(results.amplitude[target], platform, target)
    update.readout_frequency(results.frequency[target], platform, target)
    update.iq_angle(results.angle[target], platform, target)
    update.threshold(results.threshold[target], platform, target)


resonator_optimization = Protocol(
    _acquisition,
    _fit,
    _plot,
    _update,
)
"""Resonator optimization Protocol object"""
