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
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.protocols.characterization.utils import table_dict, table_html

from .utils import COLORBAND, COLORBAND_LINE, chi2_reduced


@dataclass
class FlippingParameters(Parameters):
    """Flipping runcard inputs."""

    nflips_max: int
    """Maximum number of flips ([RX(pi) - RX(pi)] sequences). """
    nflips_step: int
    """Flip step."""


@dataclass
class FlippingResults(Results):
    """Flipping outputs."""

    amplitude: dict[QubitId, tuple[float, Optional[float]]]
    """Drive amplitude for each qubit."""
    amplitude_factors: dict[QubitId, tuple[float, Optional[float]]]
    """Drive amplitude correction factor for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    chi2: dict[QubitId, tuple[float, Optional[float]]] = field(default_factory=dict)
    """Chi squared estimate mean value and error. """


FlippingType = np.dtype(
    [("flips", np.float64), ("prob", np.float64), ("error", np.float64)]
)


@dataclass
class FlippingData(Data):
    """Flipping acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    pi_pulse_amplitudes: dict[QubitId, float]
    """Pi pulse amplitudes for each qubit."""
    data: dict[QubitId, npt.NDArray[FlippingType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: FlippingParameters,
    platform: Platform,
    qubits: Qubits,
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
        pi_pulse_amplitudes={
            qubit: qubits[qubit].native_gates.RX.amplitude for qubit in qubits
        },
    )
    # sweep the parameter
    for flips in range(0, params.nflips_max, params.nflips_step):
        # create a sequence of pulses for the experiment
        sequence = PulseSequence()
        ro_pulses = {}
        for qubit in qubits:
            RX90_pulse = platform.create_RX90_pulse(qubit, start=0)
            sequence.add(RX90_pulse)
            # execute sequence RX(pi/2) - [RX(pi) - RX(pi)] from 0...flips times - RO
            start1 = RX90_pulse.duration
            for _ in range(flips):
                RX_pulse1 = platform.create_RX_pulse(qubit, start=start1)
                start2 = start1 + RX_pulse1.duration
                RX_pulse2 = platform.create_RX_pulse(qubit, start=start2)
                sequence.add(RX_pulse1)
                sequence.add(RX_pulse2)
                start1 = start2 + RX_pulse2.duration

            # add ro pulse at the end of the sequence
            ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=start1)
            sequence.add(ro_pulses[qubit])
        # execute the pulse sequence
        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.SINGLESHOT,
            ),
        )
        for qubit in qubits:
            prob = results[qubit].probability(state=1)
            data.register_qubit(
                FlippingType,
                (qubit),
                dict(
                    flips=[flips],
                    prob=prob.tolist(),
                    error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
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
    amplitude_correction_factors = {}
    chi2 = {}
    for qubit in qubits:
        qubit_data = data[qubit]
        pi_pulse_amplitude = data.pi_pulse_amplitudes[qubit]
        y = qubit_data.prob
        x = qubit_data.flips
        pguess = [0.5, 0.5, 1, np.pi, 0]
        try:
            popt, perr = curve_fit(
                flipping_fit,
                x,
                y,
                p0=pguess,
                maxfev=2000000,
                bounds=(
                    [0, 0, -np.inf, 0, 0],
                    [1, np.inf, np.inf, 2 * np.pi, np.inf],
                ),
                sigma=qubit_data.error,
            )
            perr = np.sqrt(np.diag(perr)).tolist()
            popt = popt.tolist()
        except:
            log.warning("flipping_fit: the fitting was not succesful")
            popt = [0] * 4
            perr = [1] * 4

        if popt[3] > np.pi / 2 and popt[3] < 3 * np.pi / 2:
            signed_correction = -popt[2] / 2
        else:
            signed_correction = popt[2] / 2
        # The amplitude is directly proportional to the rotation angle
        corrected_amplitudes[qubit] = (
            float((pi_pulse_amplitude * np.pi) / (np.pi + signed_correction)),
            float(
                pi_pulse_amplitude
                * np.pi
                * 1
                / (np.pi + signed_correction) ** 2
                * perr[2]
                / 2
            ),
        )
        fitted_parameters[qubit] = popt
        amplitude_correction_factors[qubit] = (
            float(signed_correction / np.pi * pi_pulse_amplitude),
            float(perr[2] * pi_pulse_amplitude / np.pi / 2),
        )
        chi2[qubit] = (
            chi2_reduced(
                y,
                flipping_fit(x, *popt),
                qubit_data.error,
            ),
            np.sqrt(2 / len(x)),
        )

    return FlippingResults(
        corrected_amplitudes, amplitude_correction_factors, fitted_parameters, chi2
    )


def _plot(data: FlippingData, qubit, fit: FlippingResults = None):
    """Plotting function for Flipping."""

    figures = []
    fig = go.Figure()

    fitting_report = ""
    qubit_data = data[qubit]

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
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                    float(fit.fitted_parameters[qubit][4]),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
        )
        fitting_report = table_html(
            table_dict(
                qubit,
                [
                    "Amplitude correction factor",
                    "Corrected amplitude [a.u.]",
                    "chi2 reduced",
                ],
                [
                    np.round(fit.amplitude_factors[qubit], 4),
                    np.round(fit.amplitude[qubit], 4),
                    fit.chi2[qubit],
                ],
                display_error=True,
            )
        )

    # last part
    fig.update_layout(
        showlegend=True,
        xaxis_title="Flips",
        yaxis_title="Probability",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: FlippingResults, platform: Platform, qubit: QubitId):
    update.drive_amplitude(results.amplitude[qubit], platform, qubit)


flipping = Routine(_acquisition, _fit, _plot, _update)
"""Flipping Routine  object."""
