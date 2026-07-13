"""Calibration for microwave crosstalk mitigation that sweeps cancelllation pulse phase."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    ParallelSweepers,
    Parameter,
    PulseSequence,
    Sweeper,
)
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

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
from qibocal.protocols.utils import table_dict, table_html
from qibocal.update import replace

__all__ = ["microwave_cross_cancellation"]


@dataclass
class MWCancellationPhaseParameters(Parameters):
    """RabiAmplitudeFreq parameters."""


MWCancPhaseType = np.dtype(
    [
        ("phase", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class MWCancellationPhaseData(Data):
    """MWCancellationPhase data acquisition."""

    cross_amplitude: dict[QubitPairId, float] = field(default_factory=dict)
    """Cross amplitude for each qubit pair."""
    data: dict[QubitPairId, npt.NDArray[MWCancPhaseType]] = field(default_factory=dict)
    """Raw data acquired."""

    def phases(self, qubit: QubitId) -> npt.NDArray:
        """Unique qubit amplitudes."""
        return np.unique(self[qubit].phase)


@dataclass
class MWCancellationPhaseResults(Results):
    """MWCancellationPhase results."""

    cancellation_pulses: dict[QubitId, dict[Literal["amplitude", "phase"], float]] = (
        field(default_factory=dict)
    )
    """Cancellation phase and amplitude for each qubit."""


def _acquisition(
    params: MWCancellationPhaseParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> MWCancellationPhaseData:
    """Data acquisition for Rabi experiment sweeping amplitude."""

    phase_range = (0, 6.4, 0.1)

    experiment_sequence = PulseSequence()
    updates = {}
    canc_pulses = []
    cross_amplitudes = {}
    for qubit, drive_line in targets:
        single_q_seq = PulseSequence()
        qubit_natives = platform.parameters.native_gates.single_qubit[qubit]
        qd_channel, qd_pulse = qubit_natives.RX()[0]
        qro_channel, qro_pulse = qubit_natives.MZ()[0]

        # retrieving the channel related to the drive line
        drive_channel, _ = platform.parameters.native_gates.single_qubit[
            drive_line
        ].RX()[0]

        # pulse amplitude from drive_line that flips qubit
        cross_ampl = platform.calibration.microwave_crosstalk_matrix[qubit, drive_line]
        # 180 amplitude when we drive qubit on its own line
        direct_ampl = platform.calibration.microwave_crosstalk_matrix[qubit, qubit]

        if not np.isfinite(cross_ampl):
            raise ValueError(
                f"Rabi amplitude calibration is missing for qubit {qubit} on drive line {drive_line}."
            )

        # creating the crosstalk pulse on line drive_line
        cross_pulse = replace(
            qd_pulse.new(),
            duration=qd_pulse.duration,
            amplitude=cross_ampl,
        )
        # creating the pulse on qubit's line with rescaled amplitude
        # in order to cancel the crosstalk one
        cancellation_pulse = replace(
            cross_pulse.new(),
            amplitude=cross_pulse.amplitude * direct_ampl / cross_ampl,
        )
        cross_amplitudes[(qubit, drive_line)] = cross_ampl

        single_q_seq |= PulseSequence(
            [
                (drive_channel, cross_pulse),
                (qd_channel, cancellation_pulse),
            ]
        )

        qubit_frequency = platform.parameters.configs[qd_channel].frequency
        updates |= {drive_channel: {"frequency": qubit_frequency}}
        canc_pulses.append(cancellation_pulse)

        single_q_seq |= PulseSequence([(qro_channel, qro_pulse)])
        experiment_sequence += single_q_seq

    phase_sweeper = Sweeper(
        parameter=Parameter.relative_phase,
        range=phase_range,
        pulses=canc_pulses,
    )

    data = MWCancellationPhaseData(
        cross_amplitude=cross_amplitudes,
    )

    results = platform.execute(
        [experiment_sequence],
        [ParallelSweepers([phase_sweeper])],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for pair in targets:
        qubit, _ = pair
        ro_pulse = list(
            experiment_sequence.channel(platform.qubits[qubit].acquisition)
        )[-1]
        prob = results[ro_pulse.id]
        data.register_qubit(
            MWCancPhaseType,
            pair,
            dict(
                phase=phase_sweeper.values,
                prob=prob.tolist(),
                error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
            ),
        )
    return data


def _fit(data: MWCancellationPhaseData) -> MWCancellationPhaseResults:
    """Do not perform any fitting procedure."""

    cancellation_pulses = {}
    for pair in data.data:
        canc_ampl = data.cross_amplitude[pair]
        phases = data.phases(pair)
        probabilities = data[pair].prob

        try:
            # creating the window for the savgol filter
            savgol_window = np.min((len(probabilities) // 10, 4))
            # defining the polynomial order for the savgol filter
            savgol_poly_order = np.clip(savgol_window // 2, 3, savgol_window).astype(
                int
            )

            first_derivative = savgol_filter(
                x=probabilities,
                window_length=savgol_window,
                polyorder=savgol_poly_order,
                deriv=1,
            )
            second_derivative = savgol_filter(
                x=probabilities,
                window_length=savgol_window,
                polyorder=savgol_poly_order,
                deriv=2,
            )

            # finding minima and maxima of the signal
            first_der_roots = CubicSpline(phases, first_derivative).roots()
            # creating a cubi spline for the second derivative of the signal
            second_derivative_spline = CubicSpline(phases, second_derivative)

            # compute the curvature for each minima and maxima
            first_der_roots_curvature = second_derivative_spline(first_der_roots)
            # we are interested only in minima, so we filter out negative curvatures
            minima_curvatures = first_der_roots_curvature[
                first_der_roots_curvature >= 0
            ]
            # we select the minimum with the smallest curvature: more stable
            optimal_minimum_idx = np.argmin(minima_curvatures)

            cancellation_pulses |= {
                pair: {
                    "phase": phases[optimal_minimum_idx] % (2 * np.pi),
                    "amplitude": canc_ampl,
                }
            }

        except Exception as e:
            log.warning(f"Rabi fit failed for pair {pair} due to {e}.")

    return MWCancellationPhaseResults(
        cancellation_pulses=cancellation_pulses,
    )


def _plot(
    data: MWCancellationPhaseData,
    target: QubitPairId,
    fit: MWCancellationPhaseResults = None,
):
    """Plotting function for RabiAmplitudeFrequency."""
    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=("Probability",),
    )
    qubit_data = data[target]
    phases = data.phases(target)

    fig.update_xaxes(title_text="Phase [rad.]", row=1, col=1)
    fig.update_yaxes(title_text="Excited State Probability", row=1, col=1)

    figures.append(fig)

    fig.add_trace(
        go.Scatter(
            x=phases,
            y=qubit_data.prob,
            mode="markers",
            error_y=dict(
                type="data",
                array=qubit_data.error,
                visible=True,
            ),
        ),
        row=1,
        col=1,
    )

    if fit is not None and target in fit.cancellation_pulses:
        fig.add_trace(
            go.Scatter(
                x=[fit.cancellation_pulses[target]["phase"]] * 2,
                y=[min(qubit_data.prob) - 0.1, max(qubit_data.prob) + 0.1],
                mode="lines",
                line=go.scatter.Line(color="orange", width=3, dash="dash"),
            ),
            row=1,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                [target] * 2,
                ["Cancellation phase", "Cancellation amplitude"],
                [
                    fit.cancellation_pulses[target]["phase"],
                    fit.cancellation_pulses[target]["amplitude"],
                ],
            )
        )

    fig.update_layout(
        showlegend=False,
        legend={"orientation": "h"},
    )
    return figures, fitting_report


def _update(
    results: MWCancellationPhaseResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    if target in results.cancellation_pulses:
        qubit, drive_line = target
        platform.calibration.set_microwave_crosstalk(
            qubit=qubit,
            microwave_line=drive_line,
            module=results.cancellation_pulses[target]["amplitude"],
            phase=-results.cancellation_pulses[target]["phase"],
        )


microwave_cross_cancellation = Protocol(_acquisition, _fit, _plot, _update)
"""Rabi amplitude with frequency tuning."""
