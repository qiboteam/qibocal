import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import plotly.graph_objects as go
import scipy.ndimage as ndimage
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
from qibocal.protocols.two_qubit_interaction.chevron.utils import COLORAXIS
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
    best_fidelity: dict[QubitId, list]
    """Best assignment fidelities."""
    fid_best_freq: dict[QubitId, float]
    """Resonator Frequency with the highest assignment fidelity."""
    fid_best_amp: dict[QubitId, list]
    """Resonator Amplitude with the highest assignment fidelity"""
    fid_best_qnd: dict[QubitId, list]
    """Quantum Non Demolition-ness with the highest assignment fidelity."""
    best_qnd: dict[QubitId, list]
    """Best quantum non demolition-ness."""
    qnd_best_freq: dict[QubitId, list]
    """Resonator Frequency with the highest quantum non demolition-ness."""
    qnd_best_amp: dict[QubitId, list]
    """Resonator Amplitude with the highest quantum non demolition-ness."""
    qnd_best_fid: dict[QubitId, list]
    """Fidelity with highest quantum non demolition-ness"""
    best_angle: dict[QubitId, float]
    """IQ angle that maximes assignment fidelity."""
    best_threshold: dict[QubitId, float]
    """Threshold that maximes assignment fidelity."""

    def __contains__(self, key):
        return True


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

    for state, pi in product([0, 1], repeat=2):
        sequence = PulseSequence()
        for qubit in targets:
            natives = platform.natives.single_qubit[qubit]
            ro_channel, ro_pulse_m1 = natives.MZ()[0]
            _, ro_pulse_m2 = natives.MZ()[0]
            if state == 1:
                sequence += natives.RX()
            sequence.append((ro_channel, Delay(duration=natives.RX()[0][1].duration)))
            sequence.append((ro_channel, ro_pulse_m1))
            sequence.append((ro_channel, Delay(duration=params.delay)))
            if pi == 1:
                sequence += natives.RX()
            sequence.append((ro_channel, ro_pulse_m2))

            freq_sweepers[qubit] = Sweeper(
                parameter=Parameter.frequency,
                values=readout_frequency(qubit, platform) + delta_frequency_range,
                channels=[platform.qubits[qubit].probe],
            )
            data.frequencies_swept[qubit] = freq_sweepers[qubit].values.tolist()

            ro_pulses_m1[qubit] = ro_pulse_m1
            ro_pulses_m2[qubit] = ro_pulse_m2

        amp_sweeper = Sweeper(
            parameter=Parameter.amplitude,
            range=(
                params.amplitude_min,
                params.amplitude_max,
                params.amplitude_step,
            ),
            pulses=[ro_pulses_m1[qubit] for qubit in targets]
            + [ro_pulses_m2[qubit] for qubit in targets],
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
                [ro_pulses_m1[target].id, ro_pulses_m2[target].id]
            ):
                data.data[target, state, m, pi] = results[pulse_id]

    return data


def _fit(data: ResonatorOptimizationData) -> ResonatorOptimizationResults:
    qubits = data.qubits
    arr = {}
    fid_best_freq = {}
    fid_best_amps = {}
    fid_best_qnd = {}
    best_angle = {}
    best_threshold = {}
    highest_fidelity = {}
    qnd_best_freq = {}
    qnd_best_amps = {}
    qnd_best_fid = {}
    best_qnd = {}

    for qubit in qubits:
        freq_vals = data.frequencies_swept[qubit]
        amp_vals = data.amplitudes_swept
        shape = (len(freq_vals), len(amp_vals))
        grid_keys = ["fidelity", "angle", "threshold", "qnd"]
        versions = ["standard", "pi"]

        grids = {
            key: {version: np.zeros(shape) for version in versions} for key in grid_keys
        }

        for j, k, pi in product(range(len(freq_vals)), range(len(amp_vals)), [0, 1]):
            version = "pi" if pi else "standard"
            iq_values = np.concatenate(
                (
                    data.data[qubit, 0, 0, pi][:, k, j, :],
                    data.data[qubit, 1, 0, pi][:, k, j, :],
                )
            )
            nshots = iq_values.shape[0] // 2
            states = [0] * nshots + [1] * nshots
            if pi:
                states.reverse()

            model = QubitFit()
            model.fit(iq_values, np.array(states))
            grids["angle"][version][j, k] = model.angle
            grids["threshold"][version][j, k] = model.threshold

            m1_state_0 = classify(
                data.data[qubit, 0, 0, pi][:, k, j, :],
                model.angle,
                model.threshold,
            )
            m1_state_1 = classify(
                data.data[qubit, 1, 0, pi][:, k, j, :],
                model.angle,
                model.threshold,
            )
            m2_state_0 = classify(
                data.data[qubit, 0, 1, pi][:, k, j, :],
                model.angle,
                model.threshold,
            )
            m2_state_1 = classify(
                data.data[qubit, 1, 1, pi][:, k, j, :],
                model.angle,
                model.threshold,
            )

            grids["fidelity"][version][j, k] = compute_assignment_fidelity(
                m1_state_1, m1_state_0
            )
            grids["qnd"][version][j, k], _, _ = compute_qnd(
                m1_state_1, m1_state_0, m2_state_1, m2_state_0
            )
            grids["angle"][version][j, k] = model.angle
            grids["threshold"][version][j, k] = model.threshold
            arr[qubit, "fidelity", version] = grids["fidelity"][version]
            arr[qubit, "qnd", version] = grids["qnd"][version]
        size = (math.ceil(len(freq_vals) / 5), math.ceil(len(amp_vals) / 5))
        filtered_fidelity = {
            version: ndimage.uniform_filter(
                grids["fidelity"][version],
                size=size,
                mode="nearest",
            )
            for version in versions
        }

        # indices for best fidelity
        i, j = np.unravel_index(
            np.argmax(filtered_fidelity["standard"]),
            filtered_fidelity["standard"].shape,
        )
        highest_fidelity[qubit] = filtered_fidelity["standard"][i, j]
        fid_best_freq[qubit] = freq_vals[i]
        fid_best_amps[qubit] = amp_vals[j]
        fid_best_qnd[qubit] = grids["qnd"]["standard"][i, j]
        best_angle[qubit] = grids["angle"]["standard"][i, j]
        best_threshold[qubit] = grids["threshold"]["standard"][i, j]

        # indices for best qnd
        i, j = np.unravel_index(
            np.argmax(grids["qnd"]["standard"]),
            grids["qnd"]["standard"].shape,
        )
        best_qnd[qubit] = grids["qnd"]["standard"][i, j]
        highest_fidelity[qubit] = filtered_fidelity["standard"][i, j]
        qnd_best_freq[qubit] = freq_vals[i]
        qnd_best_amps[qubit] = amp_vals[j]
        qnd_best_fid[qubit] = filtered_fidelity["standard"][i, j]

    return ResonatorOptimizationResults(
        data=arr,
        best_fidelity=highest_fidelity,
        fid_best_freq=fid_best_freq,
        fid_best_amp=fid_best_amps,
        fid_best_qnd=fid_best_qnd,
        best_qnd=best_qnd,
        qnd_best_freq=qnd_best_freq,
        qnd_best_amp=qnd_best_amps,
        qnd_best_fid=qnd_best_fid,
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
        rows=2,
        cols=2,
        subplot_titles=(
            "Fidelity",
            "Quantum Non Demolition-ness",
            r"Fidelity ($\pi$)",
            r"Quantum Non Demolition-ness ($\pi$)",
        ),
    )
    x, y = data.grid(target)
    if fit is not None:
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y * HZ_TO_GHZ,
                z=fit.data[target, "fidelity", "standard"],
                coloraxis=COLORAXIS[0],
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y * HZ_TO_GHZ,
                z=fit.data[target, "fidelity", "pi"],
                coloraxis=COLORAXIS[0],
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[fit.fid_best_amp[target]],
                y=[fit.fid_best_freq[target] * HZ_TO_GHZ],
                mode="markers",
                marker=dict(
                    size=8,
                    color="black",
                    symbol="cross",
                ),
                name="highest assignment fidelity",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y * HZ_TO_GHZ,
                z=fit.data[target, "qnd", "standard"],
                coloraxis=COLORAXIS[1],
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[fit.qnd_best_amp[target]],
                y=[fit.qnd_best_freq[target] * HZ_TO_GHZ],
                mode="markers",
                marker=dict(
                    size=8,
                    color="black",
                    symbol="cross",
                ),
                name="highest quantum non demolition-ness",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y * HZ_TO_GHZ,
                z=fit.data[target, "qnd", "pi"],
                coloraxis=COLORAXIS[1],
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            xaxis_title="Amplitude [a.u.]",
            xaxis2_title="Amplitude [a.u.]",
            xaxis3_title="Amplitude [a.u.]",
            xaxis4_title="Amplitude [a.u.]",
            yaxis_title="Frequency [GHz]",
            yaxis3_title="Frequency [GHz]",
            legend=dict(orientation="h"),
        )
        fig.update_layout(
            coloraxis={"colorscale": "Plasma", "colorbar": {"x": 1.15}},
            coloraxis2={"colorscale": "Viridis", "colorbar": {"x": -0.15}},
            coloraxis3={"colorscale": "Plasma", "colorbar": {"x": 1.15}},
            coloraxis4={"colorscale": "Viridis", "colorbar": {"x": -0.15}},
        )
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Best Assignment-Fidlity Amplitude [a.u.]",
                    "Best Assignment-Fidlity Frequency [GHz]",
                    "Best Assignment-Fidlity",
                    "Best QND Amplitude [a.u.]",
                    "Best QND Frequency [GHz]",
                    "Best Quantum Non Demolition-ness",
                ],
                [
                    np.round(fit.fid_best_amp[target], 4),
                    np.round(fit.fid_best_freq[target]) * HZ_TO_GHZ,
                    fit.best_fidelity[target],
                    np.round(fit.qnd_best_amp[target], 4),
                    np.round(fit.qnd_best_freq[target]) * HZ_TO_GHZ,
                    fit.best_qnd[target],
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
    update.readout_amplitude(results.fid_best_amp[target], platform, target)
    update.readout_frequency(results.fid_best_freq[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


resonator_optimization = Routine(
    _acquisition,
    _fit,
    _plot,
    _update,
)
"""Resonator optimization Routine object"""
