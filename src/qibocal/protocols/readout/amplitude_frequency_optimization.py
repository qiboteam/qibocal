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
    PulseLike,
    PulseSequence,
    Sweeper,
)

from qibocal import update
from qibocal.auto.operation import (
    Data,
    Parameters,
    Protocol,
    QubitId,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    Range,
    RangeLike,
    classify,
    compute_assignment_fidelity,
    compute_qnd,
    readout_frequency,
    table_dict,
    table_html,
    to_range,
)

from .utils import fit_classification_model

__all__ = ["ro_amplitude_frequency"]


def elaborate_raw_data(
    qubit: QubitId,
    pixel_measurements: dict[tuple[QubitId, int, int], np.ndarray],
    ampl_sweep: list[float],
    freq_sweep: list[float],
) -> dict[tuple[QubitId, str], np.ndarray]:
    """Compute readout-quality metrics over an amplitude-frequency grid.

    It returns a mapping from metric names to two-dimensional arrays indexed by
    amplitude and frequency. The computed metrics are assignment fidelity, classification
    angle and threshold, and QND fidelities.
    """

    # TODO: try to vectorize this function to avoid the for loops and speed up the computation
    shape = (len(ampl_sweep), len(freq_sweep))
    grid_keys = ["fidelity", "angle", "threshold", "qnd", "qnd-pi"]
    grids = {(qubit, key): np.zeros(shape) for key in grid_keys}
    for j, k in product(range(len(ampl_sweep)), range(len(freq_sweep))):
        measurements = {
            (m, state): pixel_measurements[qubit, state, m][:, j, k, :]
            for state in (0, 1)
            for m in range(3)
        }

        model = fit_classification_model(measurements[0, 0], measurements[0, 1])

        grids[qubit, "angle"][j, k] = model.angle
        grids[qubit, "threshold"][j, k] = model.threshold

        classified_states = {
            key: classify(val, model.angle, model.threshold)
            for key, val in measurements.items()
        }

        grids[qubit, "fidelity"][j, k] = compute_assignment_fidelity(
            classified_states[0, 1], classified_states[0, 0]
        )
        grids[qubit, "qnd"][j, k], _, _ = compute_qnd(
            classified_states[0, 1],
            classified_states[0, 0],
            classified_states[1, 1],
            classified_states[1, 0],
        )
        # for m3 we swap them because we apply a pi pulse
        grids[qubit, "qnd-pi"][j, k], _, _ = compute_qnd(
            classified_states[1, 1],
            classified_states[1, 0],
            classified_states[2, 0],
            classified_states[2, 1],
            pi=True,
        )

    return grids


@dataclass
class ReadoutAmplitudeFrequencyParameters(Parameters):
    """Resonator optimization runcard inputs"""

    frequency_range: RangeLike
    """Frequency RangeLike object; for further information, see
    :class:`qibocal.protocols.utils.RangeLike`."""
    amplitude_range: RangeLike
    """Amplitude RangeLike object; for further information, see
    :class:`qibocal.protocols.utils.RangeLike`."""
    delay: float = 0
    """Delay between readouts, could account for resonator depletion or not [ns]."""
    save_iq: bool = False
    """Whether to save the IQ data during the acquisition."""

    @property
    def _amplitude_range(self) -> Range:
        return to_range(self.amplitude_range)


@dataclass
class ReadoutAmplitudeFrequencyResults(Results):
    """Resonator optimization outputs"""

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


@dataclass
class ReadoutAmplitudeFrequencyData(Data):
    """Data class for readout optimization protocol."""

    frequencies_swept: dict[QubitId, list[float]] = field(default_factory=dict)
    """Frequency swept for each qubit."""
    amplitudes_swept: list[float] = field(default_factory=list)
    """Amplitude swept (same for all qubits)."""
    data: dict[tuple, np.ndarray] = field(default_factory=dict)
    """Raw data acquired"""

    def __contains__(self, key: QubitId | QubitPairId | tuple[QubitId, ...]) -> bool:
        """Check whether plotting data is available for qubit."""
        return all(
            (key, metric) in self.data for metric in ["fidelity", "qnd", "qnd-pi"]
        )


def _acquisition(
    params: ReadoutAmplitudeFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutAmplitudeFrequencyData:
    """Protocol to optimize readout frequency and readout amplitude.

    After preparing either state 0 or state 1 we perform two consecutive measurements to
    evaluate QND. Additionally we apply a pi pulse and we perform a third measurement to
    evaluate the QND-pi following https://arxiv.org/pdf/2110.04285"""

    ro_pulses: dict[tuple[QubitId, int, int], PulseLike] = {}
    sequences: list[PulseSequence] = []
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

    data = ReadoutAmplitudeFrequencyData()

    freq_sweepers: dict[QubitId, Sweeper] = {}
    for qubit in targets:
        freqs = to_range(
            params.frequency_range, center=readout_frequency(qubit, platform)
        )
        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            range=freqs,
            channels=[platform.qubits[qubit].probe],
        )
        data.frequencies_swept[qubit] = freq_sweepers[qubit].values.tolist()

    amp_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=params._amplitude_range,
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

    for target in targets:
        pixel_data: dict[tuple[QubitId, int, int], np.ndarray] = {}
        for state in [0, 1]:
            for m in range(3):
                pixel_data[target, state, m] = results[ro_pulses[target, state, m].id]

        data.data |= elaborate_raw_data(
            qubit=target,
            pixel_measurements=pixel_data,
            ampl_sweep=data.amplitudes_swept,
            freq_sweep=data.frequencies_swept[target],
        )

        if params.save_iq:
            data.data |= pixel_data

    return data


def _fit(data: ReadoutAmplitudeFrequencyData) -> ReadoutAmplitudeFrequencyResults:
    frequency = {}
    amplitude = {}
    angle = {}
    threshold = {}
    best_fidelity = {}
    best_qnd = {}
    best_qnd_pi = {}

    for qubit in data.qubits:
        averaged_qnd = (data.data[qubit, "qnd"] + data.data[qubit, "qnd-pi"]) / 2

        # Mask low-fidelity and invalid points before selecting the optimum.
        averaged_qnd[data.data[qubit, "fidelity"] < 0.8] = np.nan
        averaged_qnd[averaged_qnd > 1] = np.nan
        try:
            i, j = np.unravel_index(np.nanargmax(averaged_qnd), averaged_qnd.shape)
            best_fidelity[qubit] = data.data[qubit, "fidelity"][i, j]
            best_qnd[qubit] = data.data[qubit, "qnd"][i, j]
            best_qnd_pi[qubit] = data.data[qubit, "qnd-pi"][i, j]
            frequency[qubit] = data.frequencies_swept[qubit][j]
            amplitude[qubit] = data.amplitudes_swept[i]
            angle[qubit] = data.data[qubit, "angle"][i, j]
            threshold[qubit] = data.data[qubit, "threshold"][i, j]
        except ValueError:
            log.warning("Fitting error.")

    return ReadoutAmplitudeFrequencyResults(
        fidelity=best_fidelity,
        qnd=best_qnd,
        qnd_pi=best_qnd_pi,
        frequency=frequency,
        amplitude=amplitude,
        angle=angle,
        threshold=threshold,
    )


def _plot(
    data: ReadoutAmplitudeFrequencyData,
    fit: ReadoutAmplitudeFrequencyResults,
    target: QubitId,
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

    fig.add_trace(
        go.Heatmap(
            x=np.array(data.frequencies_swept[target]) * HZ_TO_GHZ,
            y=data.amplitudes_swept,
            z=data.data[target, "fidelity"],
            coloraxis="coloraxis",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=np.array(data.frequencies_swept[target]) * HZ_TO_GHZ,
            y=data.amplitudes_swept,
            z=data.data[target, "qnd"],
            coloraxis="coloraxis",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            x=np.array(data.frequencies_swept[target]) * HZ_TO_GHZ,
            y=data.amplitudes_swept,
            z=data.data[target, "qnd-pi"],
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
        coloraxis={"colorscale": "Viridis", "cmin": 0, "cmax": 1},
        legend={"orientation": "h"},
    )

    # use fit.frequency as proxy for having found a best point
    if fit is not None and target in fit:
        for col in range(1, ncols + 1):
            fig.add_trace(
                go.Scatter(
                    x=[fit.frequency[target] * HZ_TO_GHZ],
                    y=[fit.amplitude[target]],
                    mode="markers",
                    marker={"size": 8, "color": "black", "symbol": "cross"},
                    name="Best Readout Point",
                    showlegend=col == 1,
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
    results: ReadoutAmplitudeFrequencyResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.readout_amplitude(results.amplitude[target], platform, target)
    update.readout_frequency(results.frequency[target], platform, target)
    update.iq_angle(results.angle[target], platform, target)
    update.threshold(results.threshold[target], platform, target)


ro_amplitude_frequency = Protocol(_acquisition, _fit, _plot, _update)
"""Readout amplitude-frequency optimization Protocol object"""
