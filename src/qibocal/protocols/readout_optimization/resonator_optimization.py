from dataclasses import dataclass, field, fields
from itertools import product

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from ... import update
from ...auto.operation import Data, Parameters, QubitId, Results, Routine
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
    """Assignment fideilty at optimal readout point."""
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

    def __contains__(self, key: QubitId) -> bool:
        """Check whether results are available for qubit."""
        return all(
            key in k
            for k in map(
                lambda f: getattr(self, f.name),
                filter(lambda f: f.name != "data", fields(self)),
            )
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

    After preparing either state 0 or state 1 we perform two consecutive measurements to evaluate QND.
    Additionaly we apply a pi pulse and we perform a third measurement to evaluate the QND-pi
    following https://arxiv.org/pdf/2110.04285"""

    freq_sweepers = {}
    ro_pulses_m1 = {}
    ro_pulses_m2 = {}
    ro_pulses_m3 = {}

    data = ResonatorOptimizationData()

    for state in [0, 1]:
        sequence = PulseSequence()
        for qubit in targets:
            natives = platform.natives.single_qubit[qubit]
            ro_channel = platform.qubits[qubit].acquisition
            drive_channel = platform.qubits[qubit].drive
            _, ro_pulse_m1 = natives.MZ()[0]
            _, ro_pulse_m2 = natives.MZ()[0]
            _, ro_pulse_m3 = natives.MZ()[0]
            if state == 1:
                sequence += natives.RX()
                sequence.append((ro_channel, Delay(duration=natives.RX().duration)))
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
            sequence.append((ro_channel, Delay(duration=natives.RX().duration)))
            sequence.append((ro_channel, Delay(duration=params.delay)))
            sequence.append((ro_channel, ro_pulse_m3))

            freq_sweepers[qubit] = Sweeper(
                parameter=Parameter.frequency,
                values=readout_frequency(qubit, platform) + params.frequency_span,
                channels=[platform.qubits[qubit].probe],
            )
            data.frequencies_swept[qubit] = freq_sweepers[qubit].values.tolist()

            ro_pulses_m1[qubit] = ro_pulse_m1
            ro_pulses_m2[qubit] = ro_pulse_m2
            ro_pulses_m3[qubit] = ro_pulse_m3

        amp_sweeper = Sweeper(
            parameter=Parameter.amplitude,
            range=(
                params.amplitude_min,
                params.amplitude_max,
                params.amplitude_step,
            ),
            pulses=[ro_pulses_m1[qubit] for qubit in targets]
            + [ro_pulses_m2[qubit] for qubit in targets]
            + [ro_pulses_m3[qubit] for qubit in targets],
        )

        data.amplitudes_swept = amp_sweeper.values.tolist()

        results = platform.execute(
            [sequence],
            [[amp_sweeper], [freq_sweepers[q] for q in targets]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        )

        for target in targets:
            for m, pulse_id in enumerate(
                [
                    ro_pulses_m1[target].id,
                    ro_pulses_m2[target].id,
                    ro_pulses_m3[target].id,
                ]
            ):
                data.data[target, state, m] = results[pulse_id]

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
            iq_values = np.concatenate(
                (
                    data.data[qubit, 0, 0][:, j, k, :],
                    data.data[qubit, 1, 0][:, j, k, :],
                )
            )
            nshots = iq_values.shape[0] // 2
            states = [0] * nshots + [1] * nshots

            model = QubitFit()
            model.fit(iq_values, np.array(states))
            grids["angle"][j, k] = model.angle
            grids["threshold"][j, k] = model.threshold

            m1_state_0 = classify(
                data.data[qubit, 0, 0][:, j, k, :],
                model.angle,
                model.threshold,
            )

            m1_state_1 = classify(
                data.data[qubit, 1, 0][:, j, k, :],
                model.angle,
                model.threshold,
            )

            m2_state_0 = classify(
                data.data[qubit, 0, 1][:, j, k, :],
                model.angle,
                model.threshold,
            )

            m2_state_1 = classify(
                data.data[qubit, 1, 1][:, j, k, :],
                model.angle,
                model.threshold,
            )

            m3_state_0 = classify(
                data.data[qubit, 0, 2][:, j, k, :],
                model.angle,
                model.threshold,
            )

            m3_state_1 = classify(
                data.data[qubit, 1, 2][:, j, k, :],
                model.angle,
                model.threshold,
            )

            grids["fidelity"][j, k] = compute_assignment_fidelity(
                m1_state_1, m1_state_0
            )
            grids["qnd"][j, k], _, _ = compute_qnd(
                m1_state_1,
                m1_state_0,
                m2_state_1,
                m2_state_0,
            )
            # for m3 we swap them because we apply a pi pulse
            grids["qnd-pi"][j, k], _, _ = compute_qnd(
                m2_state_1, m2_state_0, m3_state_0, m3_state_1, pi=True
            )
            grids["angle"][j, k] = model.angle
            grids["threshold"][j, k] = model.threshold
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

        # Layout updates
        fig.update_layout(
            yaxis_title="Amplitude [a.u.]",
            xaxis_title="Frequency [GHz]",
            xaxis2_title="Frequency [GHz]",
            xaxis3_title="Frequency [GHz]",
            coloraxis=dict(colorscale="Viridis", cmin=0, cmax=1),
            legend=dict(orientation="h"),
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


resonator_optimization = Routine(
    _acquisition,
    _fit,
    _plot,
    _update,
)
"""Resonator optimization Routine object"""
