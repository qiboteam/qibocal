from dataclasses import dataclass, field
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

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import HZ_TO_GHZ, readout_frequency, table_dict, table_html

from ..utils import classify, compute_assignment_fidelity, compute_qnd

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


@dataclass
class ResonatorOptimizationResults(Results):
    """Resonator optimization outputs"""

    data: dict[tuple[QubitId, str, str], np.ndarray]
    best_fidelity: dict[QubitId, float]
    best_qnd: dict[QubitId, float]
    best_qnd_pi: dict[QubitId, float]
    frequency: dict[tuple[QubitId, str], float]
    """Resonator Frequency with the highest assignment fidelity."""
    amplitude: dict[tuple[QubitId, str], float]
    """Resonator Amplitude with the highest assignment fidelity"""
    angle: dict[tuple[QubitId, str], float]
    """IQ angle that maximes assignment fidelity."""
    threshold: dict[tuple[QubitId, str], float]
    """Threshold that maximes assignment fidelity."""

    def __contains__(self, key):
        return key in self.best_fidelity


@dataclass
class ResonatorOptimizationData(Data):
    """Data class for resonator optimization protocol."""

    resonator_type: str
    """Resonator type."""
    delay: float = 0
    """Delay between readouts [ns]."""
    frequencies_swept: dict[QubitId, list[float]] = field(default_factory=dict)
    amplitudes_swept: list[float] = field(default_factory=list)
    angle: dict[QubitId, float] = field(default_factory=dict)
    threshold: dict[QubitId, float] = field(default_factory=dict)
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
    freq_sweepers = {}
    ro_pulses_m1 = {}
    ro_pulses_m2 = {}
    ro_pulses_m3 = {}

    data = ResonatorOptimizationData(
        resonator_type=platform.resonator_type,
        angle={
            qubit: platform.config(platform.qubits[qubit].acquisition).iq_angle
            for qubit in targets
        },
        threshold={
            qubit: platform.config(platform.qubits[qubit].acquisition).threshold
            for qubit in targets
        },
        delay=params.delay,
    )

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
                values=readout_frequency(qubit, platform) + delta_frequency_range,
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
        shape = (len(freq_vals), len(amp_vals))
        grid_keys = ["fidelity", "angle", "threshold", "qnd", "qnd-pi"]

        grids = {key: np.zeros(shape) for key in grid_keys}

        for j, k in product(range(len(freq_vals)), range(len(amp_vals))):
            iq_values = np.concatenate(
                (
                    data.data[qubit, 0, 0][:, k, j, :],
                    data.data[qubit, 1, 0][:, k, j, :],
                )
            )
            nshots = iq_values.shape[0] // 2
            states = [0] * nshots + [1] * nshots

            model = QubitFit()
            model.fit(iq_values, np.array(states))
            grids["angle"][j, k] = model.angle
            grids["threshold"][j, k] = model.threshold

            m1_state_0 = classify(
                data.data[qubit, 0, 0][:, k, j, :],
                model.angle,
                model.threshold,
            )

            m1_state_1 = classify(
                data.data[qubit, 1, 0][:, k, j, :],
                model.angle,
                model.threshold,
            )

            m2_state_0 = classify(
                data.data[qubit, 0, 1][:, k, j, :],
                model.angle,
                model.threshold,
            )

            m2_state_1 = classify(
                data.data[qubit, 1, 1][:, k, j, :],
                model.angle,
                model.threshold,
            )

            m3_state_0 = classify(
                data.data[qubit, 0, 2][:, k, j, :],
                model.angle,
                model.threshold,
            )

            m3_state_1 = classify(
                data.data[qubit, 1, 2][:, k, j, :],
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
            grids["qnd-pi"][j, k], _, _ = compute_qnd(
                m2_state_1, m2_state_0, m3_state_0, m3_state_1, pi=True
            )
            grids["angle"][j, k] = model.angle
            grids["threshold"][j, k] = model.threshold
        arr[qubit, "fidelity"] = grids["fidelity"].copy()
        arr[qubit, "qnd"] = grids["qnd"].copy()
        arr[qubit, "qnd-pi"] = grids["qnd-pi"].copy()

        # mask values where fidelity is below 80%
        mask = grids["fidelity"] < 0.8
        for feat, value in zip(
            ["fidelity", "qnd", "qnd-pi"],
            [grids["fidelity"], grids["qnd"], grids["qnd-pi"]],
        ):
            value[mask] = np.nan
            # mask unphysical regions where QND > 1
            value[value > 1] = np.nan
            try:
                i, j = np.unravel_index(np.nanargmax(value), value.shape)
                if feat == "fidelity":
                    best_fidelity[qubit] = grids["fidelity"][i, j]
                elif feat == "qnd":
                    best_qnd[qubit] = grids["qnd"][i, j]
                else:
                    best_qnd_pi[qubit] = grids["qnd-pi"][i, j]
                frequency[qubit, feat] = freq_vals[i]
                amplitude[qubit, feat] = amp_vals[j]
                angle[qubit, feat] = grids["angle"][i, j]
                threshold[qubit, feat] = grids["threshold"][i, j]
            except ValueError:
                log.warning("Fitting error.")

    return ResonatorOptimizationResults(
        data=arr,
        best_fidelity=best_fidelity,
        best_qnd=best_qnd,
        best_qnd_pi=best_qnd_pi,
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
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Fidelity", "QND", "QND Pi"),
    )
    x, y = data.grid(target)
    if fit is not None:
        # Fidelity heatmap
        fig.add_trace(
            go.Heatmap(
                x=x * HZ_TO_GHZ,
                y=y,
                z=fit.data[target, "fidelity"].T.ravel(),
                coloraxis="coloraxis",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[fit.frequency[target, "fidelity"] * HZ_TO_GHZ],
                y=[fit.amplitude[target, "fidelity"]],
                mode="markers",
                marker=dict(size=8, color="black", symbol="cross"),
                name="highest assignment fidelity",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # QND heatmap
        fig.add_trace(
            go.Heatmap(
                x=x * HZ_TO_GHZ,
                y=y,
                z=fit.data[target, "qnd"].T.ravel(),
                coloraxis="coloraxis",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[fit.frequency[target, "qnd"] * HZ_TO_GHZ],
                y=[fit.amplitude[target, "qnd"]],
                mode="markers",
                marker=dict(size=8, color="black", symbol="cross"),
                name="highest qnd",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

        # QND-pi heatmap
        fig.add_trace(
            go.Heatmap(
                x=x * HZ_TO_GHZ,
                y=y,
                z=fit.data[target, "qnd-pi"].T.ravel(),
                coloraxis="coloraxis",
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=[fit.frequency[target, "qnd-pi"] * HZ_TO_GHZ],
                y=[fit.amplitude[target, "qnd-pi"]],
                mode="markers",
                marker=dict(size=8, color="black", symbol="cross"),
                name="qnd pi",
                showlegend=True,
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
            legend=dict(
                orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5
            ),
        )

        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Best Assignment-Fidelity",
                    "Best QND",
                    "Best QND Pi",
                ],
                [
                    np.round(fit.best_fidelity[target], 4),
                    np.round(fit.best_qnd[target], 4),
                    np.round(fit.best_qnd_pi[target], 4),
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
    pass
    # update.readout_amplitude(results.fid_best_amp[target], platform, target)
    # update.readout_frequency(results.fid_best_freq[target], platform, target)
    # update.iq_angle(results.best_angle[target], platform, target)
    # update.threshold(results.best_threshold[target], platform, target)


resonator_optimization = Routine(
    _acquisition,
    _fit,
    _plot,
    _update,
)
"""Resonator optimization Routine object"""
