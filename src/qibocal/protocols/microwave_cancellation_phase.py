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
    IqChannel,
    ParallelSweepers,
    Parameter,
    PulseSequence,
    Sweeper,
    VirtualZ,
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
from qibocal.protocols.two_qubit_interaction.cross_resonance.cr_parent_classes import (
    check_qubit_overlap,
)
from qibocal.protocols.utils import angle_wrap, table_dict, table_html
from qibocal.update import replace

__all__ = ["microwave_cross_cancellation"]


SAVGOL_WINDOW = 15
SAVGOL_POLYORDER = 3


@dataclass
class MWCancellationPhaseParameters(Parameters):
    """RabiAmplitudeFreq parameters."""

    input_drive_ampl: float | None = None
    """Input drive amplitude to use for the cancellation phase calibration."""


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

    mw_crosstalk_pulses: dict[
        QubitPairId, dict[Literal["amplitude", "phase"], float]
    ] = field(default_factory=dict)
    """Cancellation phase and amplitude for each qubit."""


def _acquisition(
    params: MWCancellationPhaseParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> MWCancellationPhaseData:
    """Data acquisition for Rabi experiment sweeping amplitude."""

    # check validity of the input
    check_qubit_overlap(targets)

    phase_range = (0, 6.4, 0.2)

    experiment_sequence = PulseSequence()
    updates = {}
    phase_pulses = []
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

        cross_channel_obj = platform.channels[drive_channel]
        qubit_channel_obj = platform.channels[qd_channel]
        if all(
            [isinstance(ch, IqChannel) for ch in [qubit_channel_obj, cross_channel_obj]]
        ):
            q_lo_params = platform.parameters.configs[qubit_channel_obj.lo]
            updates |= {
                cross_channel_obj.lo: {
                    "frequency": q_lo_params.frequency,
                    "power": q_lo_params.power,
                }
            }

        # Pi pulse amplitude from drive_line that flips qubit
        cross_ampl, _ = platform.calibration.get_microwave_crosstalk(qubit, drive_line)
        # Pi pulse amplitude when we drive qubit on its own line
        direct_ampl, _ = platform.calibration.get_microwave_crosstalk(qubit, qubit)

        # if params.input_drive_ampl is None:
        #     params.input_drive_ampl = float(np.clip(cross_ampl, -1, 1))

        # creating the crosstalk pulse on line drive_line
        cross_pulse = replace(
            qd_pulse.new(),
            duration=qd_pulse.duration,
            amplitude=float(np.clip(cross_ampl, -1, 1)),
        )
        # creating the pulse on qubit's line with rescaled amplitude
        # in order to cancel the crosstalk one
        cancellation_pulse = replace(
            cross_pulse.new(),
            amplitude=float(params.input_drive_ampl),
            # amplitude=float(cross_pulse.amplitude * direct_ampl / cross_ampl),
        )
        cross_amplitudes[(qubit, drive_line)] = cross_ampl

        # adding the phase to sweep
        phase_shift_pulse = VirtualZ(phase=0)
        single_q_seq.append((qd_channel, phase_shift_pulse))

        single_q_seq |= PulseSequence(
            [
                (drive_channel, cross_pulse),
                (qd_channel, cancellation_pulse),
            ]
        )

        qubit_frequency = platform.parameters.configs[qd_channel].frequency
        updates |= {drive_channel: {"frequency": qubit_frequency}}
        phase_pulses.append(phase_shift_pulse)

        single_q_seq |= PulseSequence([(qro_channel, qro_pulse)])
        experiment_sequence += single_q_seq

    phase_sweeper = Sweeper(
        parameter=Parameter.phase,
        range=phase_range,
        pulses=phase_pulses,
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

    mw_crosstalk_pulses = {}
    for pair in data.data:
        cross_ampl = data.cross_amplitude[pair]
        phases = data.phases(pair)
        probabilities = data[pair].prob

        try:
            if not np.isfinite(cross_ampl):
                qubit, drive_line = pair
                raise ValueError(
                    "Not enough microwave crosstalk to calibrate cancellation phase for "
                    f"qubit {qubit} on drive line {drive_line}, calibration is not updated"
                )

            first_derivative = savgol_filter(
                x=probabilities,
                window_length=SAVGOL_WINDOW,
                polyorder=SAVGOL_POLYORDER,
                deriv=1,
            )
            second_derivative = savgol_filter(
                x=probabilities,
                window_length=SAVGOL_WINDOW,
                polyorder=SAVGOL_POLYORDER,
                deriv=2,
            )

            # finding minima and maxima of the signal
            first_der_roots = CubicSpline(phases, first_derivative).roots()

            # creating a cubic spline for the second derivative of the signal
            second_derivative_spline = CubicSpline(phases, second_derivative)

            # compute the curvature for each minima and maxima
            first_der_roots_curvature = second_derivative_spline(first_der_roots)
            # mask for selecting only the minima
            signal_minima_mask = first_der_roots_curvature > 0

            # we are interested only in minima, so we filter out negative curvatures
            minima_curvatures = first_der_roots_curvature[signal_minima_mask]

            # we select the minimum with the smallest curvature: more stable
            optimal_minimum_idx = np.argmin(minima_curvatures)

            mw_crosstalk_pulses |= {
                pair: {
                    "phase": angle_wrap(
                        first_der_roots[signal_minima_mask][optimal_minimum_idx]
                    ),
                    "amplitude": cross_ampl,
                }
            }

        except Exception as e:
            log.warning(f"Rabi fit failed for pair {pair} due to {e}.")

    return MWCancellationPhaseResults(mw_crosstalk_pulses=mw_crosstalk_pulses)


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
    phases = data.phases(target)
    probabilities = data[target].prob

    fig.update_xaxes(title_text="Phase [rad.]", row=1, col=1)
    fig.update_yaxes(title_text="Excited State Probability", row=1, col=1)

    figures.append(fig)

    fig.add_trace(
        go.Scatter(
            x=phases,
            y=probabilities,
            mode="markers",
            error_y=dict(
                type="data",
                array=data[target].error,
                visible=True,
            ),
        ),
        row=1,
        col=1,
    )

    if fit is not None and target in fit.mw_crosstalk_pulses:
        phase_vec = np.linspace(min(phases), max(phases), 200)

        filtered_probs = savgol_filter(
            x=probabilities,
            window_length=SAVGOL_WINDOW,
            polyorder=SAVGOL_POLYORDER,
        )

        first = savgol_filter(
            x=probabilities,
            window_length=SAVGOL_WINDOW,
            polyorder=SAVGOL_POLYORDER,
            deriv=1,
        )

        second = savgol_filter(
            x=probabilities,
            window_length=SAVGOL_WINDOW,
            polyorder=SAVGOL_POLYORDER,
            deriv=2,
        )

        fig.add_traces(
            [
                go.Scatter(
                    x=phase_vec,
                    y=CubicSpline(phases, filtered_probs)(phase_vec),
                    mode="lines",
                    line=go.scatter.Line(color="blue"),
                ),
                go.Scatter(
                    x=phase_vec,
                    y=CubicSpline(phases, first)(phase_vec),
                    mode="lines",
                    line=go.scatter.Line(color="red"),
                ),
                go.Scatter(
                    x=phase_vec,
                    y=CubicSpline(phases, second)(phase_vec),
                    mode="lines",
                    line=go.scatter.Line(color="green"),
                ),
                go.Scatter(
                    x=[fit.mw_crosstalk_pulses[target]["phase"]] * 2,
                    y=[min(probabilities) * 0.9, max(probabilities) * 1.1],
                    mode="lines",
                    line=go.scatter.Line(color="orange", width=3, dash="dash"),
                ),
            ],
            rows=1,
            cols=1,
        )

        fitting_report = table_html(
            table_dict(
                [target] * 2,
                ["Cancellation phase", "Crosstalk amplitude"],
                [
                    fit.mw_crosstalk_pulses[target]["phase"],
                    fit.mw_crosstalk_pulses[target]["amplitude"],
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
    if target in results.mw_crosstalk_pulses:
        qubit, drive_line = target
        platform.calibration.set_microwave_crosstalk(
            qubit=qubit,
            microwave_line=drive_line,
            module=results.mw_crosstalk_pulses[target]["amplitude"],
            phase=angle_wrap(results.mw_crosstalk_pulses[target]["phase"] + np.pi),
        )


microwave_cross_cancellation = Protocol(_acquisition, _fit, _plot, _update)
"""Rabi amplitude with frequency tuning."""
