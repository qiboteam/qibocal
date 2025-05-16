from dataclasses import dataclass, field
from typing import Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, PulseSequence
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    fallback_period,
    guess_period,
    table_dict,
    table_html,
)

from ..result import probability
from .utils import COLORBAND, COLORBAND_LINE, chi2_reduced

__all__ = ["flipping"]


def flipping_sequence(
    platform: CalibrationPlatform,
    qubit: QubitId,
    delta_amplitude: float,
    flips: int,
    rx90: bool,
):
    """Pulse sequence for flipping experiment."""

    sequence = PulseSequence()
    natives = platform.natives.single_qubit[qubit]

    sequence |= natives.R(theta=np.pi / 2)

    for _ in range(flips):
        if rx90:
            qd_channel, qd_pulse = natives.RX90()[0]
        else:
            qd_channel, qd_pulse = natives.RX()[0]

        qd_detuned = update.replace(
            qd_pulse, amplitude=qd_pulse.amplitude + delta_amplitude
        )
        sequence.append((qd_channel, qd_detuned))
        sequence.append((qd_channel, qd_detuned))

        if rx90:
            sequence.append((qd_channel, qd_detuned))
            sequence.append((qd_channel, qd_detuned))

    sequence |= natives.MZ()

    return sequence


@dataclass
class FlippingParameters(Parameters):
    """Flipping runcard inputs."""

    nflips_max: int
    """Maximum number of flips ([RX(pi) - RX(pi)] sequences). """
    nflips_step: int
    """Flip step."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""
    delta_amplitude: float = 0
    """Amplitude detuning."""
    rx90: bool = False
    """Calibration of native pi pulse, if true calibrates pi/2 pulse"""


@dataclass
class FlippingResults(Results):
    """Flipping outputs."""

    amplitude: dict[QubitId, Union[float, list[float]]]
    """Drive amplitude for each qubit."""
    delta_amplitude: dict[QubitId, Union[float, list[float]]]
    """Difference in amplitude between initial value and fit."""
    delta_amplitude_detuned: dict[QubitId, Union[float, list[float]]]
    """Difference in amplitude between detuned value and fit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    rx90: bool
    """Pi or Pi_half calibration"""
    chi2: dict[QubitId, list[float]] = field(default_factory=dict)
    """Chi squared estimate mean value and error. """


FlippingType = np.dtype(
    [("flips", np.float64), ("prob", np.float64), ("error", np.float64)]
)


@dataclass
class FlippingData(Data):
    """Flipping acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    delta_amplitude: float
    """Amplitude detuning."""
    pulse_amplitudes: dict[QubitId, float]
    """Pulse amplitudes for each qubit."""
    rx90: bool
    """Pi or Pi_half calibration"""
    data: dict[QubitId, npt.NDArray[FlippingType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: FlippingParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> FlippingData:
    r"""
    Data acquisition for flipping.

    The flipping experiment correct the delta amplitude in the qubit drive pulse. We measure a qubit after applying
    a Rx(pi/2) and N flips (Rx(pi) rotations). After fitting we can obtain the delta amplitude to refine pi pulses.
    On the y axis we measure the excited state probability.

    Args:
        params (:class:`SingleShotClassificationParameters`): input parameters
        platform (:class:`CalibrationPlatform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`FlippingData`)
    """

    data = FlippingData(
        resonator_type=platform.resonator_type,
        delta_amplitude=params.delta_amplitude,
        pulse_amplitudes={
            qubit: getattr(
                platform.natives.single_qubit[qubit], "RX90" if params.rx90 else "RX"
            )[0][1].amplitude
            for qubit in targets
        },
        rx90=params.rx90,
    )

    options = {
        "nshots": params.nshots,
        "relaxation_time": params.relaxation_time,
        "acquisition_type": AcquisitionType.DISCRIMINATION,
        "averaging_mode": AveragingMode.SINGLESHOT,
    }

    sequences = []

    flips_sweep = range(0, params.nflips_max, params.nflips_step)
    for flips in flips_sweep:
        sequence = PulseSequence()
        for qubit in targets:
            sequence += flipping_sequence(
                platform=platform,
                qubit=qubit,
                delta_amplitude=params.delta_amplitude,
                flips=flips,
                rx90=params.rx90,
            )

        sequences.append(sequence)

    if params.unrolling:
        results = platform.execute(sequences, **options)
    else:
        results = [platform.execute([sequence], **options) for sequence in sequences]

    for i in range(len(sequences)):
        for qubit in targets:
            ro_pulse = list(sequences[i].channel(platform.qubits[qubit].acquisition))[
                -1
            ]
            if params.unrolling:
                result = results[ro_pulse.id]
            else:
                result = results[i][ro_pulse.id]
            prob = probability(result, state=1)
            error = np.sqrt(prob * (1 - prob) / params.nshots)
            data.register_qubit(
                FlippingType,
                (qubit),
                dict(
                    flips=np.array([flips_sweep[i]]),
                    prob=np.array([prob]),
                    error=np.array([error]),
                ),
            )
    return data


def flipping_fit(x, offset, amplitude, omega, phase, gamma):
    return np.sin(x * omega + phase) * amplitude * np.exp(-x * gamma) + offset


def _fit(data: FlippingData) -> FlippingResults:
    r"""Post-processing function for Flipping.

    The used model is

    .. math::

        y = p_0 sin\Big(\frac{2 \pi x}{p_2} + p_3\Big) + p_1.
    """
    qubits = data.qubits
    corrected_amplitudes = {}
    fitted_parameters = {}
    delta_amplitude = {}
    delta_amplitude_detuned = {}
    chi2 = {}
    for qubit in qubits:
        qubit_data = data[qubit]
        detuned_pulse_amplitude = data.pulse_amplitudes[qubit] + data.delta_amplitude
        y = qubit_data.prob
        x = qubit_data.flips

        period = fallback_period(guess_period(x, y))
        pguess = [0.5, 0.5, 2 * np.pi / period, 0, 0]

        try:
            popt, perr = curve_fit(
                flipping_fit,
                x,
                y,
                p0=pguess,
                maxfev=2000000,
                bounds=(
                    [0.4, 0.4, -np.inf, -np.pi / 4, 0],
                    [0.6, 0.6, np.inf, np.pi / 4, np.inf],
                ),
                sigma=qubit_data.error,
            )
            perr = np.sqrt(np.diag(perr)).tolist()
            popt = popt.tolist()
            correction = popt[2] / 2

            if data.rx90:
                correction /= 2

            corrected_amplitudes[qubit] = [
                float(detuned_pulse_amplitude * np.pi / (np.pi + correction)),
                float(
                    detuned_pulse_amplitude
                    * np.pi
                    * 1
                    / (np.pi + correction) ** 2
                    * perr[2]
                    / 2
                ),
            ]

            fitted_parameters[qubit] = popt

            delta_amplitude_detuned[qubit] = [
                -correction * detuned_pulse_amplitude / (np.pi + correction),
                np.abs(
                    np.pi * detuned_pulse_amplitude * np.power(np.pi + correction, -2)
                )
                * perr[2]
                / 2,
            ]
            delta_amplitude[qubit] = [
                delta_amplitude_detuned[qubit][0] + data.delta_amplitude,
                delta_amplitude_detuned[qubit][1],
            ]

            chi2[qubit] = [
                chi2_reduced(
                    y,
                    flipping_fit(x, *popt),
                    qubit_data.error,
                ),
                np.sqrt(2 / len(x)),
            ]
        except Exception as e:
            log.warning(f"Error in flipping fit for qubit {qubit} due to {e}.")

    return FlippingResults(
        corrected_amplitudes,
        delta_amplitude,
        delta_amplitude_detuned,
        fitted_parameters,
        data.rx90,
        chi2,
    )


def _plot(data: FlippingData, target: QubitId, fit: FlippingResults = None):
    """Plotting function for Flipping."""

    figures = []
    fig = go.Figure()
    fitting_report = ""
    qubit_data = data[target]

    probs = qubit_data.prob
    error_bars = qubit_data.error

    fig.add_trace(
        go.Scatter(
            x=qubit_data.flips,
            y=qubit_data.prob,
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate((qubit_data.flips, qubit_data.flips[::-1])),
            y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
            fill="toself",
            fillcolor=COLORBAND,
            line=dict(color=COLORBAND_LINE),
            showlegend=True,
            name="Errors",
        ),
    )

    if fit is not None:
        flips_range = np.linspace(
            min(qubit_data.flips),
            max(qubit_data.flips),
            2 * len(qubit_data),
        )

        fig.add_trace(
            go.Scatter(
                x=flips_range,
                y=flipping_fit(
                    flips_range,
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
                [
                    "Delta amplitude [a.u.]",
                    "Delta amplitude (with detuning) [a.u.]",
                    "Corrected amplitude [a.u.]",
                    "chi2 reduced",
                ],
                [
                    fit.delta_amplitude[target],
                    fit.delta_amplitude_detuned[target],
                    fit.amplitude[target],
                    fit.chi2[target],
                ],
                display_error=True,
            )
        )

    # last part
    fig.update_layout(
        showlegend=True,
        xaxis_title="Flips",
        yaxis_title="Excited State Probability",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: FlippingResults, platform: CalibrationPlatform, qubit: QubitId):
    update.drive_amplitude(results.amplitude[qubit], results.rx90, platform, qubit)


flipping = Routine(_acquisition, _fit, _plot, _update)
"""Flipping Routine  object."""
