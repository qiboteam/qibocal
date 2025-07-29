from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Delay, Drag, Pulse, PulseSequence
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import probability
from qibocal.update import replace

from ..utils import COLORBAND, COLORBAND_LINE, table_dict, table_html
from .drag import (
    DragTuningData,
    DragTuningParameters,
    DragTuningResults,
    DragTuningType,
)

SEQUENCES = ["YpX9", "XpY9"]
"""Sequences used to fit drag parameter."""

__all__ = ["drag_simple"]


@dataclass
class DragTuningSimpleParameters(DragTuningParameters):
    """DragTuningSimple runcard inputs."""


@dataclass
class DragTuningSimpleResults(DragTuningResults):
    """DragTuningSimple outputs."""

    def __contains__(self, key):
        return key in self.betas


@dataclass
class DragTuningSimpleData(DragTuningData):
    """DragTuningSimple acquisition outputs."""

    def __getitem__(self, key: tuple[QubitId, str]):
        qubit, setup = key
        if setup == "YpX9":
            return self.data[qubit][::2]
        return self.data[qubit][1::2]


def add_drag(pulse: Pulse, beta: float) -> Pulse:
    """Add DRAG component to Gaussian Pulse."""
    return replace(pulse, envelope=Drag(rel_sigma=pulse.envelope.rel_sigma, beta=beta))


def _acquisition(
    params: DragTuningSimpleParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> DragTuningSimpleData:
    """Acquisition function for DRAG experiments.

    We execute two sequences YpX9 and XpY9 following
    https://rsl.yale.edu/sites/default/files/2024-08/2011-RSL-Thesis-Matthew-Reed.pdf
    for different value of the DRAG parameter.
    """

    data = DragTuningSimpleData()
    beta_range = np.arange(params.beta_start, params.beta_end, params.beta_step)

    sequences, all_ro_pulses = [], []
    for beta in beta_range:
        for setup in SEQUENCES:
            sequence = PulseSequence()
            ro_pulses = {}
            for q in targets:
                natives = platform.natives.single_qubit[q]
                ro_channel, ro_pulse = natives.MZ()[0]
                qd_channel = platform.qubits[q].drive
                if setup == "YpX9":
                    ry_sequence = natives.R(phi=np.pi / 2)
                    rx90_sequence = natives.R(theta=np.pi / 2)
                    for channel, pulse in ry_sequence:
                        sequence.append((qd_channel, add_drag(pulse, beta=beta)))
                    for channel, pulse in rx90_sequence:
                        sequence.append((qd_channel, add_drag(pulse, beta=beta)))
                    sequence.append(
                        (
                            ro_channel,
                            Delay(
                                duration=rx90_sequence.duration + ry_sequence.duration
                            ),
                        )
                    )
                else:
                    _, rx = natives.RX()[0]
                    ry90_sequence = natives.R(theta=np.pi / 2, phi=np.pi / 2)
                    sequence.append((qd_channel, add_drag(rx, beta=beta)))
                    for channel, pulse in ry90_sequence:
                        sequence.append((qd_channel, add_drag(pulse, beta=beta)))
                    sequence.append(
                        (
                            ro_channel,
                            Delay(duration=rx.duration + ry90_sequence.duration),
                        )
                    )
                sequence.append((ro_channel, ro_pulse))

            sequences.append(sequence)
            all_ro_pulses.append(
                {
                    qubit: list(sequence.channel(platform.qubits[qubit].acquisition))[
                        -1
                    ]
                    for qubit in targets
                }
            )

    options = {
        "nshots": params.nshots,
        "relaxation_time": params.relaxation_time,
        "acquisition_type": AcquisitionType.DISCRIMINATION,
        "averaging_mode": AveragingMode.SINGLESHOT,
    }

    # execute the pulse sequence
    if params.unrolling:
        results = platform.execute(sequences, **options)
        for beta, ro_pulses in zip(np.repeat(beta_range, 2), all_ro_pulses):
            for qubit in targets:
                result = results[ro_pulses[qubit].id]
                prob = probability(result, state=1)
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
    else:
        for i, sequence in enumerate(sequences):
            result = platform.execute([sequence], **options)
            setup = "YpX9" if i % 2 == 2 else "XpY9"
            for qubit in targets:
                ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[
                    -1
                ]
                prob = probability(result[ro_pulse.id], state=1)
                # store the results
                data.register_qubit(
                    DragTuningType,
                    (qubit),
                    dict(
                        prob=np.array([prob]),
                        error=np.array([np.sqrt(prob * (1 - prob) / params.nshots)]),
                        beta=np.array([beta_range[i // 2]]),
                    ),
                )

    return data


def _fit(data: DragTuningSimpleData) -> DragTuningSimpleResults:
    """Post-processing for DRAG protocol.

    A linear fit is applied for the probability of both sequences.
    The optimal is determined as the point in which the two lines met.
    """
    qubits = data.qubits
    fitted_parameters = {}
    betas_optimal = {}
    for qubit in qubits:
        for setup in SEQUENCES:
            qubit_data = data[qubit, setup]
            popt, _ = curve_fit(
                f=lambda x, a, b: a * x + b,
                xdata=qubit_data.beta,
                ydata=qubit_data.prob,
                p0=[
                    (qubit_data.prob[-1] - qubit_data.prob[0])
                    / (qubit_data.beta[-1] - qubit_data.beta[0]),
                    np.mean(qubit_data.prob),
                ],
            )
            fitted_parameters[qubit, setup] = popt.tolist()
        betas_optimal[qubit] = -(
            fitted_parameters[qubit, "YpX9"][1] - fitted_parameters[qubit, "XpY9"][1]
        ) / (fitted_parameters[qubit, "YpX9"][0] - fitted_parameters[qubit, "XpY9"][0])

    return DragTuningSimpleResults(betas_optimal, fitted_parameters)


def _plot(data: DragTuningSimpleData, target: QubitId, fit: DragTuningSimpleResults):
    """Plotting function for DragTuning."""

    figures = []
    fitting_report = ""

    fig = go.Figure()
    for setup in SEQUENCES:
        qubit_data = data[target, setup]
        fig.add_trace(
            go.Scatter(
                x=qubit_data.beta,
                y=qubit_data.prob,
                opacity=1,
                mode="lines",
                name=setup,
                showlegend=True,
                legendgroup=setup,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate((qubit_data.beta, qubit_data.beta[::-1])),
                y=np.concatenate(
                    (
                        qubit_data.prob + qubit_data.error,
                        (qubit_data.prob - qubit_data.error)[::-1],
                    )
                ),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                name=setup,
                showlegend=False,
                legendgroup=setup,
            )
        )

    # # add fitting traces
    if fit is not None:
        for setup in SEQUENCES:
            qubit_data = data[target, setup]
            betas = qubit_data.beta
            beta_range = np.linspace(
                min(betas),
                max(betas),
                20,
            )

            fig.add_trace(
                go.Scatter(
                    x=beta_range,
                    y=fit.fitted_parameters[target, setup][0] * beta_range
                    + fit.fitted_parameters[target, setup][1],
                    name=f"Fit {setup}",
                    line=go.scatter.Line(dash="dot"),
                ),
            )
        fitting_report = table_html(
            table_dict(
                target,
                ["Best DRAG parameter"],
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
    results: DragTuningSimpleResults, platform: CalibrationPlatform, target: QubitId
):
    update.drag_pulse_beta(
        results.betas[target],
        platform,
        target,
    )


drag_simple = Routine(_acquisition, _fit, _plot, _update)
"""DragTuning Routine object."""
