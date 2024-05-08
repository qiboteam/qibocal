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
from scipy.signal import find_peaks

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log
from qibocal.protocols.characterization.utils import table_dict, table_html


@dataclass
class FlippingSignalParameters(Parameters):
    """Flipping runcard inputs."""

    nflips_max: int
    """Maximum number of flips ([RX(pi) - RX(pi)] sequences). """
    nflips_step: int
    """Flip step."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""


@dataclass
class FlippingSignalResults(Results):
    """Flipping outputs."""

    amplitude: dict[QubitId, tuple[float, Optional[float]]]
    """Drive amplitude for each qubit."""
    amplitude_factors: dict[QubitId, tuple[float, Optional[float]]]
    """Drive amplitude correction factor for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


FlippingType = np.dtype([("flips", np.float64), ("signal", np.float64)])


@dataclass
class FlippingSignalData(Data):
    """Flipping acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    pi_pulse_amplitudes: dict[QubitId, float]
    """Pi pulse amplitudes for each qubit."""
    data: dict[QubitId, npt.NDArray[FlippingType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: FlippingSignalParameters,
    platform: Platform,
    targets: list[QubitId],
) -> FlippingSignalData:
    r"""
    Data acquisition for flipping.

    The flipping experiment correct the delta amplitude in the qubit drive pulse. We measure a qubit after applying
    a Rx(pi/2) and N flips (Rx(pi) rotations). After fitting we can obtain the delta amplitude to refine pi pulses.

    Args:
        params (:class:`FlippingSignalParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`FlippingSignalData`)
    """

    data = FlippingSignalData(
        resonator_type=platform.resonator_type,
        pi_pulse_amplitudes={
            qubit: platform.qubits[qubit].native_gates.RX.amplitude for qubit in targets
        },
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # sweep the parameter
    sequences, all_ro_pulses = [], []
    flips_sweep = range(0, params.nflips_max, params.nflips_step)
    for flips in flips_sweep:
        # create a sequence of pulses for the experiment
        sequence = PulseSequence()
        ro_pulses = {}
        for qubit in targets:
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

        sequences.append(sequence)
        all_ro_pulses.append(ro_pulses)

    # execute the pulse sequence
    if params.unrolling:
        results = platform.execute_pulse_sequences(sequences, options)

    elif not params.unrolling:
        results = [
            platform.execute_pulse_sequence(sequence, options) for sequence in sequences
        ]

    for ig, (flips, ro_pulses) in enumerate(zip(flips_sweep, all_ro_pulses)):
        for qubit in targets:
            serial = ro_pulses[qubit].serial
            if params.unrolling:
                result = results[serial][0]
            else:
                result = results[ig][serial]
            data.register_qubit(
                FlippingType,
                (qubit),
                dict(
                    flips=np.array([flips]),
                    signal=np.array([result.magnitude]),
                ),
            )

    return data


def flipping_fit(x, offset, amplitude, omega, phase, gamma):
    return np.sin(x * omega + phase) * amplitude * np.exp(-x * gamma) + offset


def _fit(data: FlippingSignalData) -> FlippingSignalResults:
    r"""Post-processing function for Flipping.

    The used model is

    .. math::

        y = p_0 sin\Big(\frac{2 \pi x}{p_2} + p_3\Big)*\exp{-x*p4} + p_1.
    """
    qubits = data.qubits
    corrected_amplitudes = {}
    fitted_parameters = {}
    amplitude_correction_factors = {}
    for qubit in qubits:
        qubit_data = data[qubit]
        pi_pulse_amplitude = data.pi_pulse_amplitudes[qubit]
        voltages = qubit_data.signal
        flips = qubit_data.flips
        y_min = np.min(voltages)
        # Guessing period using Fourier transform
        ft = np.fft.rfft(voltages)
        # Remove the zero frequency mode
        mags = abs(ft)[1:]
        local_maxima = find_peaks(mags, height=0)
        peak_heights = local_maxima[1]["peak_heights"]
        # Select the frequency with the highest peak
        index = (
            int(local_maxima[0][np.argmax(peak_heights)] + 1)
            if len(local_maxima[0]) > 0
            else None
        )
        f = flips[index] / (flips[1] - flips[0]) if index is not None else 1
        y_max = np.max(voltages)
        x_min = np.min(flips)
        x_max = np.max(flips)
        x = (flips - x_min) * 2 * np.pi * f / (x_max - x_min)
        y = (voltages - y_min) / (y_max - y_min)

        pguess = [0.5, 0.5, 1, np.pi, 0]
        try:
            popt, _ = curve_fit(
                flipping_fit,
                x,
                y,
                p0=pguess,
                maxfev=2000000,
                bounds=(
                    [0, 0, -np.inf, 0, 0],
                    [1, np.inf, np.inf, 2 * np.pi, np.inf],
                ),
            )
        except:
            log.warning("flipping_fit: the fitting was not succesful")
            popt = [0] * 5

        translated_popt = [
            y_min + (y_max - y_min) * popt[0],
            (y_max - y_min)
            * popt[1]
            * np.exp(x_min * popt[4] * 2 * np.pi * f / (x_max - x_min)),
            popt[2] * 2 * np.pi * f / (x_max - x_min),
            popt[3] - x_min * 2 * np.pi * f / (x_max - x_min) * popt[2],
            popt[4] * 2 * np.pi * f / (x_max - x_min),
        ]
        # TODO: this might be related to the resonator type
        if popt[3] > np.pi / 2 and popt[3] < 3 * np.pi / 2:
            signed_correction = translated_popt[2] / 2
        else:
            signed_correction = -translated_popt[2] / 2
        # The amplitude is directly proportional to the rotation angle
        corrected_amplitudes[qubit] = (pi_pulse_amplitude * np.pi) / (
            np.pi + signed_correction
        )
        fitted_parameters[qubit] = translated_popt
        amplitude_correction_factors[qubit] = (
            signed_correction / np.pi * pi_pulse_amplitude
        )
    return FlippingSignalResults(
        corrected_amplitudes, amplitude_correction_factors, fitted_parameters
    )


def _plot(data: FlippingSignalData, target, fit: FlippingSignalResults = None):
    """Plotting function for Flipping."""

    figures = []
    fig = go.Figure()
    fitting_report = ""
    qubit_data = data[target]

    fig.add_trace(
        go.Scatter(
            x=qubit_data.flips,
            y=qubit_data.signal,
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
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
                ["Amplitude correction factor", "Corrected amplitude [a.u.]"],
                [
                    np.round(fit.amplitude_factors[target], 4),
                    np.round(fit.amplitude[target], 4),
                ],
            )
        )

    # last part
    fig.update_layout(
        showlegend=True,
        xaxis_title="Flips",
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: FlippingSignalResults, platform: Platform, qubit: QubitId):
    update.drive_amplitude(results.amplitude[qubit], platform, qubit)


flipping_signal = Routine(_acquisition, _fit, _plot, _update)
"""Flipping Routine  object."""
