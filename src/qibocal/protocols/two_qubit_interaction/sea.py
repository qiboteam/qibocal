from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Protocol, QubitPairId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    COLORBAND,
    COLORBAND_LINE,
    chi2_reduced,
    fallback_period,
    guess_period,
    table_dict,
    table_html,
)

__all__ = ["standard_error_amplification"]


def cz_sea_sequence(
    platform: CalibrationPlatform,
    pair: QubitPairId,
    repetitions: int,
):
    """Pulse sequence for the CZ conditional-phase standard error amplification
    (SEA) experiment.

    Args:
        platform: CalibrationPlatform
        pair: QubitPairId
        repetitions: Number of repetitions n of the CZ gate (2n CZs in total)

    For ``repetitions`` = n, 2n CZs are applied in total, at the platform's
    current (uncorrected) CZ amplitude.
    """

    qubit_a, qubit_b = pair

    natives_a = platform.natives.single_qubit[qubit_a]
    natives_b = platform.natives.single_qubit[qubit_b]
    natives_pair = platform.natives.two_qubit[pair]

    sequence = natives_a.R(theta=np.pi / 2)

    cz_channel, cz_pulse = natives_pair.CZ()[0]

    qa_channel, x_pulse_a = natives_a.RX()[0]
    qb_channel, y_pulse_b = natives_b.R(theta=np.pi, phi=np.pi / 2)[0]

    n_cz = 2 * repetitions
    for i in range(n_cz):
        sequence.extend([(cz_channel, cz_pulse)])
        if i < n_cz - 1:
            sequence.extend([(qa_channel, x_pulse_a), (qb_channel, y_pulse_b)])

    final_channel, final_pulse = natives_a.R(theta=np.pi / 2)[0]
    sequence.extend([(final_channel, final_pulse)])

    sequence |= natives_a.MZ()

    return sequence


@dataclass
class StandardErrorAmplificationParameters(Parameters):
    """CZ conditional-phase SEA runcard inputs."""

    # NOTE: missing guardrail for maximum number of repetitions vs relaxation time and nshots
    repetitions_max: int
    """Maximum number of repetitions n (2n CZs, n of them active)."""
    repetitions_step: int
    """Repetitions step."""


@dataclass
class StandardErrorAmplificationResults(Results):
    """CZ conditional-phase SEA outputs."""

    phase_error: dict[QubitPairId, float | list[float]]
    """Fitted phase error delta per active CZ [rad]."""
    fitted_parameters: dict[QubitPairId, list[float]]
    """Raw fitting output."""
    chi2: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Chi squared estimate mean value and error."""


StandardErrorAmplificationType = np.dtype(
    [("repetitions", np.float64), ("prob", np.float64), ("error", np.float64)]
)


@dataclass
class StandardErrorAmplificationData(Data):
    """CZ conditional-phase SEA acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[QubitPairId, npt.NDArray[StandardErrorAmplificationType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _acquisition(
    params: StandardErrorAmplificationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> StandardErrorAmplificationData:
    r"""
    Data acquisition for the CZ conditional-phase SEA experiment.

    Args:
        params: StandardErrorAmplificationParameters
        platform: CalibrationPlatform
        targets: list of QubitPairId
    """

    data = StandardErrorAmplificationData(resonator_type=platform.resonator_type)

    sequences: list[PulseSequence] = []
    repetitions_sweep = range(0, params.repetitions_max, params.repetitions_step)
    for repetitions in repetitions_sweep:
        sequence = PulseSequence()
        for pair in targets:
            sequence += cz_sea_sequence(
                platform=platform,
                pair=pair,
                repetitions=repetitions,
            )
        sequences.append(sequence)

    results = platform.execute(
        sequences,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )

    for repetitions, sequence in zip(repetitions_sweep, sequences):
        for pair in targets:
            probe_qubit = pair[0]
            acq_channel = platform.qubits[probe_qubit].acquisition
            ro_pulse = list(sequence.channel(acq_channel))[-1]
            prob = results[ro_pulse.id]
            error = np.sqrt(prob * (1 - prob) / params.nshots)
            data.register_qubit(
                StandardErrorAmplificationType,
                pair,
                {
                    "repetitions": np.array([repetitions]),
                    "prob": np.array([prob]),
                    "error": np.array([error]),
                },
            )
    return data


def sea_fit(x, offset, amplitude, omega, phase, gamma):
    return np.sin(x * omega + phase) * amplitude * np.exp(-x * gamma) + offset


def _fit(data: StandardErrorAmplificationData) -> StandardErrorAmplificationResults:
    r"""Post-processing function for the CZ conditional-phase SEA experiment."""
    pairs = data.qubits
    phase_error = {}
    fitted_parameters = {}
    chi2 = {}
    for pair in pairs:
        pair_data = data[pair]
        y = pair_data["prob"]
        x = pair_data["repetitions"]

        period = fallback_period(guess_period(x, y))
        pguess = [0.5, 0.5, 2 * np.pi / period, 0, 0]

        try:
            popt, perr = curve_fit(
                sea_fit,
                x,
                y,
                p0=pguess,
                maxfev=2000000,
                bounds=(
                    [0.4, 0.4, -np.inf, -np.pi / 4, 0],
                    [0.6, 0.6, np.inf, np.pi / 4, np.inf],
                ),
                sigma=pair_data["error"],
            )
            perr = np.sqrt(np.diag(perr)).tolist()
            popt = popt.tolist()

            fitted_parameters[pair] = popt
            phase_error[pair] = [popt[2], perr[2]]

            chi2[pair] = [
                chi2_reduced(
                    y,
                    sea_fit(x, *popt),
                    pair_data["error"],
                ),
                np.sqrt(2 / len(x)),
            ]
        except Exception as e:
            log.warning(
                f"Error in CZ conditional-phase SEA fit for pair {pair} due to {e}."
            )

    return StandardErrorAmplificationResults(phase_error, fitted_parameters, chi2)


def _plot(
    data: StandardErrorAmplificationData,
    target: QubitPairId,
    fit: StandardErrorAmplificationResults | None = None,
):
    """Plotting function for the CZ conditional-phase SEA experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""
    pair_data = data[target]

    probs = pair_data["prob"]
    error_bars = pair_data["error"]

    fig.add_trace(
        go.Scatter(
            x=pair_data["repetitions"],
            y=pair_data["prob"],
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate(
                (pair_data["repetitions"], pair_data["repetitions"][::-1])
            ),
            y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
            fill="toself",
            fillcolor=COLORBAND,
            line=dict(color=COLORBAND_LINE),
            showlegend=True,
            name="Errors",
        ),
    )

    if fit is not None and target in fit.fitted_parameters:
        rep_range = np.linspace(
            min(pair_data["repetitions"]),
            max(pair_data["repetitions"]),
            2 * len(pair_data),
        )

        fig.add_trace(
            go.Scatter(
                x=rep_range,
                y=sea_fit(
                    rep_range,
                    float(fit.fitted_parameters[target][0]),
                    float(fit.fitted_parameters[target][1]),
                    float(fit.fitted_parameters[target][2]),
                    float(fit.fitted_parameters[target][3]),
                    float(fit.fitted_parameters[target][4]),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
        )

        fitting_report = table_html(
            table_dict(
                target,
                ["Phase error delta [rad]", "chi2 reduced"],
                [fit.phase_error[target], fit.chi2[target]],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Repetitions (n)",
        yaxis_title="Excited State Probability (probe qubit)",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: StandardErrorAmplificationResults,
    platform: CalibrationPlatform,
    pair: QubitPairId,
):
    """Store the estimated conditional-phase error in calibration."""
    target = tuple(pair)
    platform.calibration.two_qubits[target].conditional_phase = results.phase_error[
        target
    ][0]


standard_error_amplification = Protocol(_acquisition, _fit, _plot, _update)
