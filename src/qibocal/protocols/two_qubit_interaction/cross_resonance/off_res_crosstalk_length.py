from dataclasses import dataclass, field
from typing import Optional

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
from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitPairId,
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.rabi import utils as rabi_utils
from qibocal.protocols.utils import (
    COLORBAND,
    COLORBAND_LINE,
    chi2_reduced,
    fallback_period,
    guess_period,
    table_dict,
    table_html,
)

__all__ = ["cr_crosstalk_length"]


@dataclass
class CrCrosstalkLengthParameters(Parameters):
    """CrCrosstalkLength runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    pulse_amplitude: Optional[float] = None
    """Pi pulse amplitude. Same for all qubits."""
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""
    off_resonance: bool = True
    """Whether we drive the control qubit with its frequency or with the target qubit frequency."""


@dataclass
class CrCrosstalkLengthResults(Results):
    """CrCrosstalkLength outputs."""

    length: dict[QubitPairId, int | list[float]]
    """Pi pulse duration for each qubit."""
    amplitude: dict[QubitPairId, float | list[float]]
    """Pi pulse amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitPairId, dict[str, float]]
    """Raw fitting output."""
    chi2: dict[QubitPairId, list[float]] = field(default_factory=dict)


CrCrosstalkLenType = np.dtype(
    [("length", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthData(Data):
    """CrCrosstalkLength acquisition outputs."""

    amplitudes: dict[QubitPairId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitPairId, npt.NDArray[CrCrosstalkLenType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _acquisition(
    params: CrCrosstalkLengthParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> RabiLengthData:
    r"""
    Data acquisition for Cross Resonance Crosstalk experiment sweeping duration.
    This experiment simply consist in performing a Rabi experiment on a qubit by
    applying a pulse at the frequency of the qubit on the adjacent connected qubit.
    In this experiment then we see how driving a qbuit affect its neighbor qubit,
    and we can extract the crosstalk strength.
    """

    sequence = PulseSequence()
    qd_pulses = {}
    delays = {}
    ro_pulses = {}
    amplitudes = {}
    for pair in targets:
        control_q = pair[0]
        target_q = pair[1]
        control_natives = platform.natives.single_qubit[control_q]
        target_natives = platform.natives.single_qubit[target_q]

        qd_channel, qd_pulse = control_natives.RX()[0]
        ro_channel, ro_pulse = target_natives.MZ()[0]

        if params.pulse_amplitude is not None:
            qd_pulse = update.replace(qd_pulse, amplitude=params.pulse_amplitude)

        if params.off_resonance:
            cr_channel = platform.qubits[control_q].drive_extra[target_q]
        else:
            cr_channel = qd_channel

        amplitudes[pair] = qd_pulse.amplitude
        ro_pulses[pair] = ro_pulse
        qd_pulses[pair] = qd_pulse

        sequence.append((cr_channel, qd_pulse))
        if params.interpolated_sweeper:
            sequence.align([cr_channel, ro_channel])
        else:
            delays[pair] = Delay(duration=16)
            sequence.append((ro_channel, delays[pair]))
        sequence.append((ro_channel, ro_pulse))

    sweep_range = (
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )
    if params.interpolated_sweeper:
        sweeper = Sweeper(
            parameter=Parameter.duration_interpolated,
            range=sweep_range,
            pulses=[qd_pulses[p] for p in targets],
        )
    else:
        sweeper = Sweeper(
            parameter=Parameter.duration,
            range=sweep_range,
            pulses=([qd_pulses[p] for p in targets] + [delays[p] for p in targets]),
        )

    data = RabiLengthData(
        amplitudes=amplitudes,
    )

    # execute the sweep
    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for pair in targets:
        prob = results[ro_pulses[pair].id]
        data.register_qubit(
            CrCrosstalkLenType,
            (pair),
            dict(
                length=sweeper.values,
                prob=prob,
                error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
            ),
        )
    return data


def _fit(data: RabiLengthData) -> CrCrosstalkLengthResults:
    """Post-processing for CrCrosstalkLength experiment."""

    pairs = data.pairs
    fitted_parameters = {}
    durations = {}
    amplitudes = {}
    chi2 = {}

    for pair in pairs:
        pair_data = data[pair]
        raw_x = pair_data.length
        min_x = np.min(raw_x)
        max_x = np.max(raw_x)
        y = pair_data.prob
        x = (raw_x - min_x) / (max_x - min_x)

        period = fallback_period(guess_period(x, y))
        median_sig = np.median(y)
        q80 = np.quantile(y, 0.8)
        q20 = np.quantile(y, 0.2)
        amplitude_guess = abs(q80 - q20) / 1.5
        phase_guess = np.pi

        pguess = [median_sig, amplitude_guess, period, phase_guess, 0]

        try:
            popt, perr, pi_pulse_parameter = rabi_utils.fit_length_function(
                x,
                y,
                pguess,
                signal=False,
                x_limits=(min_x, max_x),
            )
            durations[pair] = [pi_pulse_parameter, perr[2] * (max_x - min_x) / 2]
            fitted_parameters[pair] = popt
            amplitudes = {key: [value, 0] for key, value in data.amplitudes.items()}
            chi2[pair] = [
                chi2_reduced(
                    y,
                    rabi_utils.rabi_length_function(raw_x, *popt),
                    pair_data.error,
                ),
                np.sqrt(2 / len(y)),
            ]
        except Exception as e:
            log.warning(f"Rabi fit failed for pair {pair} due to {e}.")

    return CrCrosstalkLengthResults(durations, amplitudes, fitted_parameters, chi2)


def _plot(data: RabiLengthData, fit: CrCrosstalkLengthResults, target: QubitPairId):
    """Plotting function for CrCrosstalkLength experiment."""

    figures = []
    fitting_report = ""

    qubit_data = data[target]
    probs = qubit_data.prob
    error_bars = qubit_data.error
    rabi_parameters = getattr(qubit_data, "length")
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
                y=rabi_utils.rabi_length_function(rabi_parameter_range, *params),
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
            xaxis_title="Time [ns]",
            yaxis_title="Excited state probability",
        )

    figures.append(fig)

    return figures, fitting_report


cr_crosstalk_length = Routine(
    _acquisition,
    _fit,
    _plot,
)
"""CrCrosstalkLength Routine object."""

"""See for an example of CrCrosstalkLength experiment executed on the simulator:
    http://login.qrccluster.com:9000/Nt3r6j2ITFCvjRKqHwpeBg==
"""
