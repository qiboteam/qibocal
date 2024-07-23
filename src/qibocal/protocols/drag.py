from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log

from .utils import (
    COLORBAND,
    COLORBAND_LINE,
    HZ_TO_GHZ,
    chi2_reduced,
    fallback_period,
    guess_period,
    table_dict,
    table_html,
)

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
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""


@dataclass
class DragTuningResults(Results):
    """DragTuning outputs."""

    betas: dict[QubitId, float]
    """Optimal beta paramter for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    chi2: dict[QubitId, tuple[float, Optional[float]]] = field(default_factory=dict)
    """Chi2 calculation."""


DragTuningType = np.dtype(
    [("prob", np.float64), ("error", np.float64), ("beta", np.float64)]
)


@dataclass
class DragTuningData(Data):
    """DragTuning acquisition outputs."""

    anharmonicity: dict[QubitId, float] = field(default_factory=dict)
    """Anharmonicity of each qubit."""
    data: dict[QubitId, npt.NDArray[DragTuningType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: DragTuningParameters,
    platform: Platform,
    targets: list[QubitId],
) -> DragTuningData:
    r"""
    Data acquisition for drag pulse tuning experiment.
    See https://arxiv.org/pdf/1504.06597.pdf Fig. 2 (c).
    """

    data = DragTuningData(
        anharmonicity={
            qubit: platform.qubits[qubit].anharmonicity * HZ_TO_GHZ for qubit in targets
        }
    )
    # define the parameter to sweep and its range:
    # qubit drive DRAG pulse beta parameter
    beta_param_range = np.arange(params.beta_start, params.beta_end, params.beta_step)

    sequences, all_ro_pulses = [], []
    for beta_param in beta_param_range:
        sequence = PulseSequence()
        ro_pulses = {}
        for qubit in targets:
            RX_drag_pulse = platform.create_RX_drag_pulse(
                qubit, start=0, beta=beta_param / data.anharmonicity[qubit]
            )
            RX_drag_pulse_minus = platform.create_RX_drag_pulse(
                qubit,
                start=RX_drag_pulse.finish,
                beta=beta_param / data.anharmonicity[qubit],
                relative_phase=np.pi,
            )
            ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                qubit, start=RX_drag_pulse_minus.finish
            )

            sequence.add(RX_drag_pulse)
            sequence.add(RX_drag_pulse_minus)
            sequence.add(ro_pulses[qubit])
        sequences.append(sequence)
        all_ro_pulses.append(ro_pulses)

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )
    # execute the pulse sequence
    if params.unrolling:
        results = platform.execute_pulse_sequences(sequences, options)

    elif not params.unrolling:
        results = [
            platform.execute_pulse_sequence(sequence, options) for sequence in sequences
        ]

    for ig, (beta, ro_pulses) in enumerate(zip(beta_param_range, all_ro_pulses)):
        for qubit in targets:
            serial = ro_pulses[qubit].serial
            if params.unrolling:
                result = results[serial][ig]
            else:
                result = results[ig][serial]
            prob = result.probability(state=0)
            # store the results
            data.register_qubit(
                DragTuningType,
                (qubit),
                dict(
                    prob=np.array([prob]),
                    error=np.array([np.sqrt(prob * (1 - prob) / params.nshots)]),
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
        beta_params = qubit_data.beta
        prob = qubit_data.prob

        beta_min = np.min(beta_params)
        beta_max = np.max(beta_params)
        normalized_beta = (beta_params - beta_min) / (beta_max - beta_min)

        # Guessing period using fourier transform
        period = fallback_period(guess_period(normalized_beta, prob))
        pguess = [0.5, 0.5, period, 0]

        try:
            popt, _ = curve_fit(
                drag_fit,
                normalized_beta,
                prob,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi],
                    [1, 1, np.inf, np.pi],
                ),
                sigma=qubit_data.error,
            )
            translated_popt = [
                popt[0],
                popt[1],
                popt[2] * (beta_max - beta_min),
                popt[3] - 2 * np.pi * beta_min / popt[2] / (beta_max - beta_min),
            ]
            fitted_parameters[qubit] = translated_popt
            predicted_prob = drag_fit(beta_params, *translated_popt)
            betas_optimal[qubit] = beta_params[np.argmax(predicted_prob)]
            chi2[qubit] = (
                chi2_reduced(
                    prob,
                    predicted_prob,
                    qubit_data.error,
                ),
                np.sqrt(2 / len(prob)),
            )
        except Exception as e:
            log.warning(f"drag_tuning_fit failed for qubit {qubit} due to {e}.")
    return DragTuningResults(betas_optimal, fitted_parameters, chi2=chi2)


def _plot(data: DragTuningData, target: QubitId, fit: DragTuningResults):
    """Plotting function for DragTuning."""

    figures = []
    fitting_report = ""

    qubit_data = data[target]
    betas = qubit_data.beta
    fig = go.Figure(
        [
            go.Scatter(
                x=qubit_data.beta,
                y=qubit_data.prob,
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
                        qubit_data.prob + qubit_data.error,
                        (qubit_data.prob - qubit_data.error)[::-1],
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
                ["Optimal Beta Param", "Chi2 reduced"],
                [(np.round(fit.betas[target], 4), 0), fit.chi2[target]],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Beta parameter",
        yaxis_title="Ground State Probability",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: DragTuningResults, platform: Platform, target: QubitId):
    try:
        update.drag_pulse_beta(
            results.betas[target] / platform.qubits[target].anharmonicity / HZ_TO_GHZ,
            platform,
            target,
        )
    except ZeroDivisionError:
        log.warning(
            f"Beta parameter cannot be updated since the anharmoncity for qubit {target} is 0."
        )


drag_tuning = Routine(_acquisition, _fit, _plot, _update)
"""DragTuning Routine object."""
