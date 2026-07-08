from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, PulseSequence, Readout
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitPairId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    fallback_period,
    guess_period,
    table_dict,
    table_html,
    chi2_reduced,
    COLORBAND,
    COLORBAND_LINE,
)

__all__ = ["cz_amplitude_sea"]


def cz_sea_sequence(
    platform: CalibrationPlatform,
    pair: QubitPairId,
    delta_amplitude: float,
    repetitions: int,
):
    """Pulse sequence for the CZ amplitude standard error amplification (SEA) experiment.
    
    Args:
        platform: CalibrationPlatform
        pair: QubitPairId
        delta_amplitude: CZ amplitude detuning applied during acquisition.
        repetitions: Number of repetitions n of the CZ gate (2n CZs in total)

    For ``repetitions`` = n, 2n CZs are applied in total.
    """

    qubit_a, qubit_b = pair

    natives_a = platform.natives.single_qubit[qubit_a]
    natives_b = platform.natives.single_qubit[qubit_b]
    natives_pair = platform.natives.two_qubit[pair]

    sequence = natives_a.R(theta=np.pi / 2)

    cz_channel, cz_pulse = natives_pair.CZ()[0]
    cz_detuned = update.replace(
        cz_pulse, amplitude=cz_pulse.amplitude + delta_amplitude
    )

    qa_channel, x_pulse_a = natives_a.RX()[0]
    qb_channel, y_pulse_b = natives_b.R(theta=np.pi, phi=np.pi / 2)[0]

    n_cz = 2 * repetitions
    for i in range(n_cz):
        sequence.extend([(cz_channel, cz_detuned)])
        if i < n_cz - 1:
            sequence.extend([(qa_channel, x_pulse_a), (qb_channel, y_pulse_b)])

    final_channel, final_pulse = natives_a.R(theta=np.pi / 2)[0]
    sequence.extend([(final_channel, final_pulse)])

    sequence |= natives_a.MZ()

    return sequence


@dataclass
class CZAmplitudeSEAParameters(Parameters):
    """CZ amplitude SEA runcard inputs."""

    repetitions_max: int
    """Maximum number of repetitions n (2n CZs, n of them active)."""
    repetitions_step: int
    """Repetitions step."""
    delta_amplitude: float = 0
    """CZ amplitude detuning applied during acquisition."""
    phase_amplitude_slope: float = None
    """d(phase)/d(amplitude) [rad / a.u.] of the CZ conditional phase near the
    operating point, e.g. obtained from a chevron / virtual-Z scan. Required
    to convert the measured phase error into a corrected CZ amplitude"""
    # TODO: I didn't implement the protocol for the slope


@dataclass
class CZAmplitudeSEAResults(Results):
    """CZ amplitude SEA outputs."""

    amplitude: dict[QubitPairId, float | list[float]]
    """Corrected CZ amplitude for each pair (only if phase_amplitude_slope given)."""
    phase_error: dict[QubitPairId, float | list[float]]
    """Fitted phase error delta per active CZ [rad]."""
    delta_amplitude: dict[QubitPairId, float | list[float]]
    """Amplitude correction implied by phase_error / phase_amplitude_slope."""
    fitted_parameters: dict[QubitPairId, list[float]]
    """Raw fitting output."""
    chi2: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Chi squared estimate mean value and error."""


CZAmplitudeSEAType = np.dtype(
    [("repetitions", np.float64), ("prob", np.float64), ("error", np.float64)]
)


@dataclass
class CZAmplitudeSEAData(Data):
    """CZ amplitude SEA acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    delta_amplitude: float
    """CZ amplitude detuning used during acquisition."""
    cz_amplitudes: dict[QubitPairId, float]
    """Nominal (undetuned) CZ amplitude for each pair."""
    phase_amplitude_slope: float | None = None
    """d(phase)/d(amplitude) used for the amplitude correction, if provided."""
    data: dict[QubitPairId, npt.NDArray[CZAmplitudeSEAType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _acquisition(
    params: CZAmplitudeSEAParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CZAmplitudeSEAData:
    r"""
    Data acquisition for the CZ amplitude SEA experiment.

    Args:
        params: CZAmplitudeSEAParameters
        platform: CalibrationPlatform
        targets: list of QubitPairId
    """

    data = CZAmplitudeSEAData(
        resonator_type=platform.resonator_type,
        delta_amplitude=params.delta_amplitude,
        cz_amplitudes={
            pair: platform.natives.two_qubit[pair].CZ()[0][1].amplitude
            for pair in targets
        },
        phase_amplitude_slope=params.phase_amplitude_slope,
    )

    sequences: list[PulseSequence] = []
    repetitions_sweep = range(0, params.repetitions_max, params.repetitions_step)
    for repetitions in repetitions_sweep:
        sequence = PulseSequence()
        for pair in targets:
            sequence += cz_sea_sequence(
                platform=platform,
                pair=pair,
                delta_amplitude=params.delta_amplitude,
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
                CZAmplitudeSEAType,
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


def _fit(data: CZAmplitudeSEAData) -> CZAmplitudeSEAResults:
    r"""Post-processing function for the CZ amplitude SEA experiment.
    """
    pairs = data.qubits
    corrected_amplitudes = {}
    phase_error = {}
    fitted_parameters = {}
    delta_amplitude = {}
    chi2 = {}
    for pair in pairs:
        pair_data = data[pair]
        nominal_amplitude = data.cz_amplitudes[pair] + data.delta_amplitude
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

            delta = popt[2]
            fitted_parameters[pair] = popt
            phase_error[pair] = [delta, perr[2]]

            if data.phase_amplitude_slope:
                slope = data.phase_amplitude_slope
                amp_correction = -delta / slope
                amp_correction_error = np.abs(perr[2] / slope)

                delta_amplitude[pair] = [amp_correction, amp_correction_error]
                corrected_amplitudes[pair] = [
                    float(nominal_amplitude + amp_correction),
                    float(amp_correction_error),
                ]
            else:
                log.warning(
                    f"No phase_amplitude_slope provided for pair {pair}: "
                    "reporting phase error only, no amplitude correction computed."
                )

            chi2[pair] = [
                chi2_reduced(
                    y,
                    sea_fit(x, *popt),
                    pair_data["error"],
                ),
                np.sqrt(2 / len(x)),
            ]
        except Exception as e:
            log.warning(f"Error in CZ amplitude SEA fit for pair {pair} due to {e}.")

    return CZAmplitudeSEAResults(
        corrected_amplitudes,
        phase_error,
        delta_amplitude,
        fitted_parameters,
        chi2,
    )


def _plot(
    data: CZAmplitudeSEAData,
    target: QubitPairId,
    fit: CZAmplitudeSEAResults | None = None,
):
    """Plotting function for the CZ amplitude SEA experiment."""

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

        columns = ["Phase error delta [rad]", "chi2 reduced"]
        values = [fit.phase_error[target], fit.chi2[target]]
        if target in fit.amplitude:
            columns = [
                "Phase error delta [rad]",
                "Delta amplitude [a.u.]",
                "Corrected CZ amplitude [a.u.]",
                "chi2 reduced",
            ]
            values = [
                fit.phase_error[target],
                fit.delta_amplitude[target],
                fit.amplitude[target],
                fit.chi2[target],
            ]

        fitting_report = table_html(
            table_dict(
                target,
                columns,
                values,
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
    results: CZAmplitudeSEAResults, platform: CalibrationPlatform, pair: QubitPairId
):
    """Write CZ amplitude correction in calibration."""
    target = tuple(pair)
    platform.calibration.two_qubits[target].conditional_phase = results.phase_error[target][0]

cz_amplitude_sea = Protocol(_acquisition, _fit, _plot, _update)
"""CZ amplitude SEA Protocol object."""
