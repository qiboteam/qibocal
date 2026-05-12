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

    beta: tuple[float, float, float] | None = None
    """Tuple of the beta parameters in the form: (start, stop, step)."""
    beta_start: float | None = None
    """DRAG pulse beta start sweep parameter."""
    beta_end: float | None = None
    """DRAG pulse beta end sweep parameter."""
    beta_step: float | None = None
    """DRAG pulse beta sweep step parameter."""
    nflips: int = 1
    """Repetitions of (Xpi - Xmpi)."""

    @property
    def beta_range(self) -> tuple[float, float, float]:
        """
        Return a tuple with the beta sweep (start, end, step).
        """
        has_beta = self.beta is not None
        beta_fields = [self.beta_start, self.beta_end, self.beta_step]
        has_any_beta_field = any(x is not None for x in beta_fields)
        has_all_beta_fields = all(x is not None for x in beta_fields)

        if has_any_beta_field and not has_all_beta_fields:
            raise ValueError(
                "If any of `beta_start`, `beta_end`, `beta_step` is set, all must be "
                "set."
            )
        if has_beta and has_all_beta_fields:
            raise ValueError(
                "Define either `beta` tuple or all of `beta_start`, `beta_end`, "
                "`beta_step`, but not both."
            )
        if has_beta:
            return self.beta
        if has_all_beta_fields:
            return (self.beta_start, self.beta_end, self.beta_step)
        raise ValueError(
            "Must define either `beta` tuple or all of `beta_start`, `beta_end`, "
            "`beta_step`."
        )


@dataclass
class DragTuningResults(Results):
    """DragTuning outputs."""

    betas: dict[QubitId, float]
    """Optimal beta paramter for each qubit."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output as lists for JSON serialization."""
    chi2: dict[QubitId, tuple[float, float | None]] = field(default_factory=dict)
    """Chi2 calculation."""
    # The chi2 is not included in the report, but we store it at least for now.


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
    beta_param_range = np.arange(*params.beta_range)

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
            excited_state_prob = results[ro_pulses[qubit].id]
            # store the results
            data.register_qubit(
                DragTuningType,
                (qubit),
                dict(
                    prob=np.array([excited_state_prob]),
                    error=np.array(
                        [
                            np.sqrt(
                                excited_state_prob
                                * (1 - excited_state_prob)
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

        # normalize beta
        beta_params = qubit_data["beta"]

        # Guessing period using fourier transform
        period = fallback_period(guess_period(beta_params, qubit_data["prob"]))
        pguess = [0.5, 0.5, period, 0]
        try:
            popt, _ = curve_fit(
                drag_fit,
                beta_params,
                qubit_data["prob"],
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.inf],
                    [1, 1, np.inf, np.inf],
                ),
                sigma=qubit_data["error"],
            )
            fitted_parameters[qubit] = popt.tolist()  # must be a list for JSON
            period_fit = popt[2]
            phase_fit = popt[3]

            # calculate the smallest and largest k for which the minimum lies in the
            # beta interval. Minima of drag_fit(x, offset, amplitude, period, phase)
            # occur for x = period * (k + 1/2 - phase / (2*pi)), for integer k.
            phase_2pi = phase_fit / (2 * np.pi)
            beta_min = np.min(beta_params)
            beta_max = np.max(beta_params)
            k_min = np.ceil(beta_min / period_fit + phase_2pi - 0.5).astype(int)
            k_max = np.floor(beta_max / period_fit + phase_2pi - 0.5).astype(int)
            if k_min <= k_max:
                # Choose beta value with the smallest absolute value that falls inside
                # the beta interval.
                candidate_ks = np.arange(k_min, k_max + 1)
                candidate_betas = [
                    period_fit * (k + 0.5 - phase_2pi) for k in candidate_ks
                ]
                betas_optimal[qubit] = min(candidate_betas, key=abs)
            else:
                # If no analytical minimum lies in the beta interval, minimum is
                # fixed at one of the interval boundaries.
                left_value = drag_fit(beta_min, *popt)
                right_value = drag_fit(beta_max, *popt)
                betas_optimal[qubit] = (
                    beta_min if left_value <= right_value else beta_max
                )

            predicted_prob = drag_fit(beta_params, *popt)
            chi2[qubit] = (
                chi2_reduced(
                    qubit_data["prob"],
                    predicted_prob,
                    qubit_data["error"],
                ),
                np.sqrt(2 / len(qubit_data["prob"])),
            )
        except Exception as e:
            log.warning(f"drag_tuning_fit failed for qubit {qubit} due to {e}.")
    return DragTuningResults(betas_optimal, fitted_parameters, chi2=chi2)


def _plot(
    data: DragTuningData, target: QubitId, fit: DragTuningResults
) -> tuple[list[go.Figure], str]:
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
                error_y=dict(
                    type="data",
                    array=qubit_data["error"],
                    visible=True,
                ),
                mode="markers",
                name="Data",
                showlegend=True,
                legendgroup="Probability",
            ),
        ]
    )

    # add fitting traces
    if fit is not None:
        beta_range = np.linspace(
            min(betas),
            max(betas),
            100,
        )

        fig.add_trace(
            go.Scatter(
                x=beta_range,
                y=drag_fit(beta_range, *fit.fitted_parameters[target]),
                name="Fit",
            ),
        )
        fitting_report = table_html(
            table_dict(
                target,
                ["Beta"],
                [np.round(fit.betas[target], 4)],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Beta parameter",
        yaxis_title="Excited state probability",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: DragTuningResults, platform: CalibrationPlatform, target: QubitId
) -> None:
    update.drag_pulse_beta(
        results.betas[target],
        platform,
        target,
    )


drag_tuning = Routine(_acquisition, _fit, _plot, _update)
"""DragTuning Routine object."""
