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

    def __post_init__(self):
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
        raise ValueError(
            "Must define either `beta` tuple or all of `beta_start`, `beta_end`, "
            "`beta_step`."
        )

    @property
    def beta_range(self) -> tuple[float, float, float]:
        """Return a tuple with the beta sweep (start, end, step)."""
        if self.beta is not None:
            return self.beta
        return (self.beta_start, self.beta_end, self.beta_step)


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


def drag_fit(beta, offset, amplitude, period, phase):
    return offset + amplitude * np.cos(2 * np.pi * beta / period + phase)


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
        period_guess = fallback_period(guess_period(beta_params, qubit_data["prob"]))
        pguess = [0.5, 0.5, period_guess, 0]
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
            _offset, amplitude, period, phase = popt

            # Evaluate drag_fit on a dense grid and select the minimum closest to
            # beta=0. A penalty term breaks equality between equally deep minima of the
            # sinusoidal fit and favours those points closest to beta=0.
            #
            # The penalty factor has to exceed the variation in drag_fit between
            # adjacent grid points near a minimum, which from a Taylor expansion is
            # (2*pi/period* beta_step)**2. The O(beta_step**4) term is negative so the
            # quadratic bound is safe.
            sampling_points = 1000
            beta_grid = np.linspace(
                beta_params.min(), beta_params.max(), sampling_points
            )
            beta_step = (beta_params.max() - beta_params.min()) / sampling_points
            # beta_step is divided by two because the worst-case scenario is where a
            # minimum is in the middle between two sampled points
            penalty_factor = (
                amplitude * 0.5 * (2 * np.pi / period * (beta_step / 2)) ** 2
            )
            penalty = penalty_factor * np.abs(
                np.floor(beta_grid / period + phase / (2 * np.pi))
            )
            betas_optimal[qubit] = beta_grid[
                np.argmin(drag_fit(beta_grid, *popt) + penalty)
            ]

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
