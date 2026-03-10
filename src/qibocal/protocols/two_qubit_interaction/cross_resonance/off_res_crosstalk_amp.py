from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.rabi import utils as rabi_utils
from qibocal.result import probability

from ...utils import (
    COLORBAND,
    COLORBAND_LINE,
    chi2_reduced,
    fallback_period,
    guess_period,
    table_dict,
    table_html,
)

__all__ = ["cr_crosstalk_amplitude"]


@dataclass
class CrCrosstalkAmplitudeParameters(Parameters):
    """CrCrosstalkAmplitude runcard inputs."""

    min_amp: float
    """Minimum amplitude."""
    max_amp: float
    """Maximum amplitude."""
    step_amp: float
    """Step amplitude."""
    pulse_length: float | None = None
    """RX pulse duration [ns]."""
    off_resonance: bool = True
    """Whether we drive the control qubit with its frequency or with the target qubit frequency."""


@dataclass
class CrCrosstalkAmplitudeResults(Results):
    """CrCrosstalkAmplitude outputs."""

    amplitude: dict[QubitPairId, float | list[float]]
    """Drive amplitude for each qubit."""
    length: dict[QubitPairId, float | list[float]]
    """Drive pulse duration. Same for all qubits."""
    fitted_parameters: dict[QubitPairId, dict[str, float]]
    """Raw fitted parameters."""
    chi2: dict[QubitPairId, list[float]] = field(default_factory=dict)


CrCrosstalkAmpType = np.dtype(
    [("amp", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for cross resonance crosstalk amplitude."""


@dataclass
class CrCrosstalkAmplitudeData(Data):
    """CrCrosstalkAmplitude data acquisition."""

    durations: dict[QubitPairId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitPairId, npt.NDArray[CrCrosstalkAmpType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _acquisition(
    params: CrCrosstalkAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CrCrosstalkAmplitudeData:
    r"""
    Data acquisition for Cross Resonance Crosstalk experiment sweeping amplitude.
    This experiment simply consist in performing a Rabi experiment on a qubit by
    applying a pulse at the frequency of the qubit on the adjacent connected qubit.
    In this experiment then we see how driving a qbuit affect its neighbor qubit,
    and we can extract the crosstalk strength.
    """

    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    durations = {}

    for pair in targets:
        target_q = pair[1]
        control_q = pair[0]

        control_natives = platform.natives.single_qubit[control_q]
        target_natives = platform.natives.single_qubit[target_q]

        qd_channel, qd_pulse = control_natives.RX()[0]
        ro_channel, ro_pulse = target_natives.MZ()[0]

        # #####################################
        # # test
        # target_channel, target_pulse = target_natives.RX()[0]
        # #####################################

        if params.off_resonance:
            cr_channel = platform.qubits[control_q].drive_extra[target_q]
        else:
            cr_channel = qd_channel

        if params.pulse_length is not None:
            qd_pulse = update.replace(qd_pulse, duration=params.pulse_length)

        durations[pair] = qd_pulse.duration
        qd_pulses[pair] = qd_pulse
        ro_pulses[pair] = ro_pulse

        sequence.append((cr_channel, qd_pulses[pair]))
        # ##################################################
        # # test
        # sequence.append((target_channel, Delay(duration=durations[pair])))
        # sequence.append((target_channel, target_pulse))
        # sequence.append((target_channel, Delay(duration=durations[pair])))
        # sequence.append((ro_channel, Delay(duration=target_pulse.duration)))
        # sequence.append((ro_channel, Delay(duration=durations[pair])))
        # ##################################################
        sequence.append((ro_channel, Delay(duration=durations[pair])))
        sequence.append((ro_channel, ro_pulse))

    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.min_amp, params.max_amp, params.step_amp),
        pulses=[qd_pulses[pair] for pair in targets],
    )

    data = CrCrosstalkAmplitudeData(durations=durations)

    # sweep the parameter
    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )
    for pair in targets:
        prob = probability(results[ro_pulses[pair].id], state=1)
        data.register_qubit(
            CrCrosstalkAmpType,
            (pair),
            dict(
                amp=sweeper.values,
                prob=prob.tolist(),
                error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
            ),
        )
    return data


def _fit(data: CrCrosstalkAmplitudeData) -> CrCrosstalkAmplitudeResults:
    """Post-processing for CrCrosstalkAmplitude."""
    pairs = data.pairs

    pi_pulse_amplitudes = {}
    fitted_parameters = {}
    durations = {}
    chi2 = {}

    for pair in pairs:
        pair_data = data[pair]

        x = pair_data.amp
        y = pair_data.prob

        period = fallback_period(guess_period(x, y))
        median_sig = np.median(y)
        q80 = np.quantile(y, 0.8)
        q20 = np.quantile(y, 0.2)
        amplitude_guess = abs(q80 - q20) / 1.5
        phase_guess = np.pi
        pguess = [median_sig, amplitude_guess, period, phase_guess]

        try:
            popt, perr, pi_pulse_parameter = rabi_utils.fit_amplitude_function(
                x,
                y,
                pguess,
                signal=False,
            )
            pi_pulse_amplitudes[pair] = [pi_pulse_parameter, perr[2] / 2]
            fitted_parameters[pair] = popt
            durations = {key: [value, 0] for key, value in data.durations.items()}
            chi2[pair] = [
                chi2_reduced(
                    y,
                    rabi_utils.rabi_amplitude_function(x, *popt),
                    pair_data.error,
                ),
                np.sqrt(2 / len(y)),
            ]

        except Exception as e:
            log.warning(f"Rabi fit failed for pair {pair} due to {e}.")
    return CrCrosstalkAmplitudeResults(
        pi_pulse_amplitudes, durations, fitted_parameters, chi2
    )


def _plot(
    data: CrCrosstalkAmplitudeData,
    target: QubitPairId,
    fit: CrCrosstalkAmplitudeResults = None,
):
    """Plotting function for CrCrosstalkAmplitude."""

    figures = []
    fitting_report = ""

    qubit_data = data[target]
    probs = qubit_data.prob
    error_bars = qubit_data.error
    rabi_parameters = getattr(qubit_data, "amp")
    fig = go.Figure(
        [
            go.Scatter(
                x=rabi_parameters,
                y=qubit_data.prob,
                opacity=1,
                name="Probability",
                showlegend=True,
                legendgroup="Probability",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((rabi_parameters, rabi_parameters[::-1])),
                y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    if fit is not None:
        rabi_parameter_range = np.linspace(
            min(rabi_parameters),
            max(rabi_parameters),
            2 * len(rabi_parameters),
        )
        params = fit.fitted_parameters[target]
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=rabi_utils.rabi_amplitude_function(rabi_parameter_range, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
        )

        fitting_report = table_html(
            table_dict(
                3 * [target],
                [
                    "Pi pulse amplitude [a.u.]",
                    "Pi pulse length [ns]",
                    "chi2 reduced",
                ],
                [
                    fit.amplitude[target],
                    fit.length[target],
                    fit.chi2[target],
                ],
                display_error=True,
            )
        )

        fig.update_layout(
            showlegend=True,
            xaxis_title="Amplitude [dimensionless]",
            yaxis_title="Excited state probability",
        )

    figures.append(fig)

    return figures, fitting_report


cr_crosstalk_amplitude = Routine(
    _acquisition,
    _fit,
    _plot,
)
"""CrCrosstalkAmplitude Routine object."""

"""See for an example of CrCrosstalkAmplitude experiment executed on the simulator:
    http://login.qrccluster.com:9000/00xqkYftTkaKTRGtjIlnRw==
"""
