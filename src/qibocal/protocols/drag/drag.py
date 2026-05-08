import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Drag,
    Pulse,
    PulseSequence,
    Readout,
)
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ..utils import (
    COLORBAND,
    COLORBAND_LINE,
    chi2_reduced,
    fallback_period,
    guess_period,
    table_dict,
    table_html,
)

__all__ = [
    "drag_tuning",
    "DragTuningType",
    "DragTuningParameters",
    "DragTuningResults",
    "DragTuningData",
]


# TODO: add errors in fitting
@dataclass
class DragTuningParameters(Parameters):
    """DragTuning runcard inputs."""

    beta_start: float
    """DRAG pulse beta start sweep parameter."""
    beta_end: float
    """DRAG pulse beta end sweep parameter."""
    beta_step: float
    """DRAG pulse beta sweep step parameter."""
    nflips: int = 1
    """Repetitions of (Xpi - Xmpi)."""


@dataclass
class DragTuningResults(Results):
    """DragTuning outputs."""

    betas: dict[QubitId, float]
    """Optimal beta paramter for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    chi2: dict[QubitId, tuple[float, float | None]] = field(default_factory=dict)
    """Chi2 calculation."""


DragTuningType = np.dtype(
    [("prob", np.float64), ("error", np.float64), ("beta", np.float64)]
)


@dataclass
class DragTuningData(Data):
    """DragTuning acquisition outputs."""

    data: dict[QubitId, npt.NDArray[DragTuningType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: DragTuningParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> DragTuningData:
    r"""
    Data acquisition for drag pulse tuning experiment.
    See https://arxiv.org/pdf/1504.06597.pdf Fig. 2 (c).
    """

    data = DragTuningData()
    beta_param_range = np.arange(
        params.beta_start, params.beta_end, params.beta_step
    ).tolist()

    sequences, all_ro_pulses = [], []
    for beta_param in beta_param_range:
        sequence = PulseSequence()
        ro_pulses = {}
        for q in targets:
            natives = platform.natives.single_qubit[q]
            qd_channel, qd_pulse = natives.RX()[0]
            ro_channel, ro_pulse = natives.MZ()[0]
            assert isinstance(qd_pulse, Pulse) and isinstance(qd_pulse.envelope, Drag)
            drag = qd_pulse.model_copy(
                update={
                    "envelope": Drag(
                        rel_sigma=qd_pulse.envelope.rel_sigma,
                        beta=beta_param,
                    )
                }
            )
            drag_negative = drag.model_copy(update={"relative_phase": np.pi})

            for _ in range(params.nflips):
                sequence.append((qd_channel, drag))
                sequence.append((qd_channel, drag_negative))
            sequence.append(
                (
                    ro_channel,
                    Delay(
                        duration=params.nflips
                        * (drag.duration + drag_negative.duration)
                    ),
                )
            )
            sequence.append((ro_channel, ro_pulse))
        sequences.append(sequence)
        for qubit in targets:
            acq_channel = platform.qubits[qubit].acquisition
            assert acq_channel is not None
            ro_pulse = list(sequence.channel(acq_channel))[-1]
            assert isinstance(ro_pulse, Readout)
            ro_pulses[qubit] = ro_pulse
        all_ro_pulses.append(ro_pulses)

    # execute the pulse sequences
    results = platform.execute(
        sequences,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for beta, ro_pulses in zip(beta_param_range, all_ro_pulses):
        for qubit in targets:
            ground_state_prob = 1 - results[ro_pulses[qubit].id]
            # store the results
            data.register_qubit(
                DragTuningType,
                (qubit),
                dict(
                    prob=np.array([ground_state_prob]),
                    error=np.array(
                        [
                            np.sqrt(
                                ground_state_prob
                                * (1 - ground_state_prob)
                                / params.nshots
                            )
                        ]
                    ),
                    beta=np.array([beta]),
                ),
            )

    return data


def drag_fit(x, offset, amplitude, period, phase):
    return offset + amplitude * np.cos(2 * np.pi * x / period + phase)


def _fit(data: DragTuningData) -> DragTuningResults:
    qubits = data.qubits
    betas_optimal = {}
    fitted_parameters = {}
    chi2 = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        # normalize prob
        prob = qubit_data["prob"]
        prob_min = np.min(prob)
        prob_max = np.max(prob)
        normalized_prob = (prob - prob_min) / (prob_max - prob_min)

        # normalize beta
        beta_params = qubit_data["beta"]
        beta_min = np.min(beta_params)
        beta_max = np.max(beta_params)
        normalized_beta = (beta_params - beta_min) / (beta_max - beta_min)

        # Guessing period using fourier transform
        period = fallback_period(guess_period(normalized_beta, normalized_prob))
        pguess = [0.5, 0.5, period, 0]
        try:
            popt, _ = curve_fit(
                drag_fit,
                normalized_beta,
                normalized_prob,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi],
                    [1, 1, np.inf, np.pi],
                ),
                sigma=qubit_data["error"],
            )
            translated_popt = [
                popt[0] * (prob_max - prob_min) + prob_min,
                popt[1] * (prob_max - prob_min),
                popt[2] * (beta_max - beta_min),
                popt[3] - 2 * np.pi * beta_min / popt[2] / (beta_max - beta_min),
            ]
            fitted_parameters[qubit] = translated_popt
            predicted_prob = drag_fit(beta_params, *translated_popt)

            period_fit = translated_popt[2]
            phase_fit = translated_popt[3]

            # calculate the smallest and largest k for which the maximum lies in the
            # beta interval. Using that maxima of drag_fit(x, offset, amplitude, period,
            # phase) occur for x = period * (k - phase / (2*pi)), for integer k.
            phase_2pi = phase_fit / (2 * np.pi)
            k_min = math.ceil(beta_min / period_fit + phase_2pi)
            k_max = math.floor(beta_max / period_fit + phase_2pi)

            if k_min <= k_max:
                # Choose beta value with the smallest absolute value that falls inside
                # the beta interval.
                candidate_ks = np.arange(k_min, k_max + 1)
                candidate_betas = [period_fit * (k - phase_2pi) for k in candidate_ks]
                betas_optimal[qubit] = min(candidate_betas, key=abs)
            else:
                # If no analytical maximum lies in the beta interval, maximum is
                # fixed at one of the interval boundaries. Bounds during the fit
                # mean
                left_value = drag_fit(beta_min, *translated_popt)
                right_value = drag_fit(beta_max, *translated_popt)
                betas_optimal[qubit] = (
                    beta_min if left_value >= right_value else beta_max
                )

            chi2[qubit] = (
                chi2_reduced(
                    prob,
                    predicted_prob,
                    qubit_data["error"],
                ),
                np.sqrt(2 / len(prob)),
            )
        except Exception as e:
            log.warning(f"drag_tuning_fit failed for qubit {qubit} due to {e}.")
    return DragTuningResults(betas_optimal, fitted_parameters, chi2=chi2[0])


def _plot(data: DragTuningData, target: QubitId, fit: DragTuningResults):
    """Plotting function for DragTuning."""

    figures = []
    fitting_report = ""

    qubit_data = data[target]
    betas = qubit_data["beta"]
    fig = go.Figure(
        [
            go.Scatter(
                x=qubit_data["beta"],
                y=qubit_data["prob"],
                opacity=1,
                mode="lines",
                name="Probability",
                showlegend=True,
                legendgroup="Probability",
            ),
            go.Scatter(
                x=np.concatenate((betas, betas[::-1])),
                y=np.concatenate(
                    (
                        qubit_data["prob"] + qubit_data["error"],
                        (qubit_data["prob"] - qubit_data["error"])[::-1],
                    )
                ),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    # add fitting traces
    if fit is not None:
        beta_range = np.linspace(
            min(betas),
            max(betas),
            20,
        )

        fig.add_trace(
            go.Scatter(
                x=beta_range,
                y=drag_fit(beta_range, *fit.fitted_parameters[target]),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
        )
        fitting_report = table_html(
            table_dict(
                target,
                ["Beta", "Chi2 reduced"],
                [np.round(fit.betas[target], 4), fit.chi2[target]],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Beta parameter",
        yaxis_title="Ground State Probability",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: DragTuningResults, platform: CalibrationPlatform, target: QubitId):
    update.drag_pulse_beta(
        results.betas[target],
        platform,
        target,
    )


drag_tuning = Routine(_acquisition, _fit, _plot, _update)
"""DragTuning Routine object."""
