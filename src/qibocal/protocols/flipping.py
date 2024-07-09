from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from scipy.optimize import curve_fit

from qibocal.auto.operation import Routine
from qibocal.config import log
from qibocal.protocols.utils import (
    fallback_period,
    guess_period,
    table_dict,
    table_html,
)

from .flipping_signal import (
    FlippingSignalData,
    FlippingSignalParameters,
    FlippingSignalResults,
    _update,
    flipping_fit,
    flipping_sequence,
)
from .utils import COLORBAND, COLORBAND_LINE, chi2_reduced


@dataclass
class FlippingParameters(FlippingSignalParameters):
    """Flipping runcard inputs."""


@dataclass
class FlippingResults(FlippingSignalResults):
    """Flipping outputs."""

    chi2: dict[QubitId, list[float]] = field(default_factory=dict)
    """Chi squared estimate mean value and error. """


FlippingType = np.dtype(
    [("flips", np.float64), ("prob", np.float64), ("error", np.float64)]
)


@dataclass
class FlippingData(FlippingSignalData):
    """Flipping acquisition outputs."""

    data: dict[QubitId, npt.NDArray[FlippingType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: FlippingParameters,
    platform: Platform,
    targets: list[QubitId],
) -> FlippingData:
    r"""
    Data acquisition for flipping.

    The flipping experiment correct the delta amplitude in the qubit drive pulse. We measure a qubit after applying
    a Rx(pi/2) and N flips (Rx(pi) rotations). After fitting we can obtain the delta amplitude to refine pi pulses.

    Args:
        params (:class:`SingleShotClassificationParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`FlippingData`)
    """

    data = FlippingData(
        resonator_type=platform.resonator_type,
        delta_amplitude=params.delta_amplitude,
        pi_pulse_amplitudes={
            qubit: platform.qubits[qubit].native_gates.RX.amplitude for qubit in targets
        },
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    # sweep the parameter
    sequences, all_ro_pulses = [], []
    flips_sweep = range(0, params.nflips_max, params.nflips_step)
    for flips in flips_sweep:
        # create a sequence of pulses for the experiment
        sequence = PulseSequence()
        for qubit in targets:
            sequence += flipping_sequence(
                platform=platform,
                qubit=qubit,
                delta_amplitude=params.delta_amplitude,
                flips=flips,
            )

        sequences.append(sequence)
        all_ro_pulses.append(sequence.ro_pulses)

    # execute the pulse sequence
    if params.unrolling:
        results = platform.execute_pulse_sequences(sequences, options)

    elif not params.unrolling:
        results = [
            platform.execute_pulse_sequence(sequence, options) for sequence in sequences
        ]

    for ig, (flips, ro_pulses) in enumerate(zip(flips_sweep, all_ro_pulses)):
        for qubit in targets:
            serial = ro_pulses.get_qubit_pulses(qubit)[0].serial
            if params.unrolling:
                result = results[serial][0]
            else:
                result = results[ig][serial]
            prob = result.probability(state=1)
            error = np.sqrt(prob * (1 - prob) / params.nshots)
            data.register_qubit(
                FlippingType,
                (qubit),
                dict(
                    flips=np.array([flips]),
                    prob=np.array([prob]),
                    error=np.array([error]),
                ),
            )

    return data


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
        detuned_pi_pulse_amplitude = (
            data.pi_pulse_amplitudes[qubit] + data.delta_amplitude
        )
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
            corrected_amplitudes[qubit] = [
                float(detuned_pi_pulse_amplitude * np.pi / (np.pi + correction)),
                float(
                    detuned_pi_pulse_amplitude
                    * np.pi
                    * 1
                    / (np.pi + correction) ** 2
                    * perr[2]
                    / 2
                ),
            ]

            fitted_parameters[qubit] = popt

            delta_amplitude_detuned[qubit] = [
                -correction * detuned_pi_pulse_amplitude / (np.pi + correction),
                np.abs(
                    np.pi
                    * detuned_pi_pulse_amplitude
                    * np.power(np.pi + correction, -2)
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


flipping = Routine(_acquisition, _fit, _plot, _update)
"""Flipping Routine  object."""
