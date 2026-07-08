from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.constants
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)
from qibolab._core.components import IqChannel
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    lorentzian,
    table_dict,
    table_html,
)
from qibocal.result import magnitude, phase

__all__ = [
    "qubit_spectroscopy",
    "QubitSpectroscopyParameters",
    "QubitSpectroscopyResults",
    "QubitSpectroscopyData",
    "_fit",
]


QubitSpecType = np.dtype(
    [
        ("freq", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for qubit spectroscopy."""


@dataclass
class QubitSpectroscopyParameters(Parameters):
    """QubitSpectroscopy runcard inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative  to the qubit frequency."""
    freq_step: int
    """Frequency [Hz] step for sweep."""
    drive_duration: int
    """Drive pulse duration [ns]. Same for all qubits."""
    drive_amplitude: float | None = None
    """Drive pulse amplitude (optional). Same for all qubits."""


@dataclass
class QubitSpectroscopyResults(Results):
    """QubitSpectroscopy outputs."""

    frequency: dict[QubitId, dict[str, float]]
    """Drive frequecy [GHz] for each qubit."""
    amplitude: dict[QubitId, float]
    """Input drive amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""


@dataclass
class QubitSpectroscopyData(Data):
    """QubitSpectroscopy acquisition outputs."""

    amplitudes: dict[QubitId, float]
    """Amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _calculate_batches(freq_width: int, max_if_bandwidth: int = 300_000_000):
    """
    Calculate frequency batches for wideband spectroscopy.

    """
    batch_starts = np.arange(-freq_width / 2, freq_width / 2, 2 * max_if_bandwidth)
    batch_ends = np.append(batch_starts[1:], freq_width / 2)
    batch_limits = np.stack((batch_starts, batch_ends))
    lo_offsets = batch_limits.sum(axis=0) / 2 if len(batch_starts) > 1 else [0]
    return np.vstack((batch_limits, lo_offsets)).T


def _acquisition(
    params: QubitSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitSpectroscopyData:
    """Data acquisition for qubit spectroscopy.

    Handles wideband spectroscopy by batching when the frequency range exceeds ±300 MHz from the LO
    """

    # Calculate batches
    batches = _calculate_batches(params.freq_width)

    # Get drive channels and LO channels for each qubit
    drive_channels = {}
    lo_channels = {}
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel, _ = natives.RX()[0]
        drive_channels[qubit] = qd_channel

        # Get the LO channel associated with this drive channel
        channel_obj = platform.channels[qd_channel]
        if isinstance(channel_obj, IqChannel) and channel_obj.lo is not None:
            lo_channels[qubit] = channel_obj.lo
        else:
            lo_channels[qubit] = None

    # Initialize storage for intermediate results
    values = {qubit: defaultdict(list) for qubit in targets}
    drive_amplitudes: dict[QubitId, float] = {}

    # Execute each batch
    for start, end, lo_offset in batches:
        delta_frequency_range = np.arange(start, end, params.freq_step)

        # Build the pulse sequence
        sequence = PulseSequence()
        ro_pulses = {}
        sweepers = []

        for qubit in targets:
            natives = platform.natives.single_qubit[qubit]
            qd_channel = drive_channels[qubit]
            _, qd_pulse = natives.RX()[0]
            ro_channel, ro_pulse = natives.MZ()[0]

            qd_pulse = update.replace(qd_pulse, duration=params.drive_duration)
            if params.drive_amplitude is not None:
                qd_pulse = update.replace(qd_pulse, amplitude=params.drive_amplitude)

            if qubit not in drive_amplitudes:
                # If already added, skip re-adding since the drive amplitude is
                # unchanged across batches.
                drive_amplitudes[qubit] = qd_pulse.amplitude

            ro_pulses[qubit] = ro_pulse

            sequence.append((qd_channel, qd_pulse))
            sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
            sequence.append((ro_channel, ro_pulse))

            f0 = platform.config(qd_channel).frequency
            sweepers.append(
                Sweeper(
                    parameter=Parameter.frequency,
                    values=f0 + delta_frequency_range,
                    channels=[qd_channel],
                )
            )

        # Prepare updates for this batch
        batch_updates = []
        for qubit in targets:
            update_dict = {}

            # Update the frequency of the drive channel to avoid raising a validation an error
            update_dict[drive_channels[qubit]] = {
                "frequency": platform.config(drive_channels[qubit]).frequency
                + lo_offset
            }

            # If we're batching, update the LO
            if lo_offset != 0 and lo_channels[qubit] is not None:
                f0 = platform.config(drive_channels[qubit]).frequency
                update_dict[lo_channels[qubit]] = {"frequency": f0 + lo_offset}

            batch_updates.append(update_dict)

        # Execute this batch
        results = platform.execute(
            [sequence],
            [sweepers],
            updates=batch_updates,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
        )

        # Collect results from this batch
        for qubit in targets:
            result = results[ro_pulses[qubit].id]
            f0 = platform.config(drive_channels[qubit]).frequency

            signal = magnitude(result)
            _phase = phase(result)

            # Store results with absolute frequencies
            values[qubit]["frequency"].append(delta_frequency_range + f0)
            values[qubit]["signal"].append(signal)
            values[qubit]["phase"].append(_phase)

    # Create data structure and aggregate results
    data = QubitSpectroscopyData(amplitudes=drive_amplitudes)

    # Combine all batches for each qubit
    for qubit in targets:
        # Concatenate arrays from all batches
        freq = np.concatenate(values[qubit]["frequency"])
        signal = np.concatenate(values[qubit]["signal"])
        _phase = np.concatenate(values[qubit]["phase"])

        data.register_qubit(
            QubitSpecType,
            (qubit),
            dict(
                signal=signal,
                phase=_phase,
                freq=freq,
            ),
        )

    return data


def _lorentzian_with_offset(frequency, amplitude, center, sigma, offset):
    return lorentzian(frequency, amplitude, center, sigma) + offset


def _guess_initial_parameters(signal, frequencies, is_peak):
    k = max(1, int(0.1 * len(signal)))
    guess_offset = np.mean(np.concatenate([signal[:k], signal[-k:]]))

    guess_offset = (signal[0] + signal[-1]) / 2
    signal_no_background = signal - guess_offset

    sign = 1 if is_peak else -1
    signal_flipped = signal_no_background * sign
    guess_center = frequencies[np.argmax(signal_flipped)]
    guess_peak_height = signal_flipped.max() * sign
    indices_beyond_half = np.where(signal_flipped > signal_flipped.max() / 2)[0]

    if len(indices_beyond_half) >= 1:
        guess_sigma = (
            frequencies[indices_beyond_half[-1]] - frequencies[indices_beyond_half[0]]
        ) / 2
    else:
        # if there is no clear peak, we give a high flexibility
        guess_sigma = frequencies[-1] - frequencies[0]

    guess_amp = guess_peak_height * guess_sigma * np.pi

    return [
        guess_amp,
        guess_center,
        guess_sigma,
        guess_offset,
    ]


def _lorentzian_fit(data):
    frequencies = data.freq
    signal = data.signal

    freq_domain_size = frequencies[-1] - frequencies[0]

    # Try to fit both peak and dip, and pick the best one
    best_fit = None
    for is_peak in [True, False]:
        initial_parameters = _guess_initial_parameters(signal, frequencies, is_peak)
        bounds = (
            [0.0 if is_peak else -np.inf, frequencies[0], 0.0, -np.inf],
            [np.inf if is_peak else 0.0, frequencies[-1], freq_domain_size, np.inf],
        )

        # fit the model with the data and guessed parameters
        try:
            fit_parameters, parameters_cov = curve_fit(
                _lorentzian_with_offset,
                frequencies,
                signal,
                p0=initial_parameters,
                bounds=bounds,
            )
        except RuntimeError:
            continue

        sum_sq_residuals = float(
            np.sum(
                (signal - _lorentzian_with_offset(frequencies, *fit_parameters)) ** 2
            )
        )

        if best_fit is None or sum_sq_residuals < best_fit["sum_sq_residuals"]:
            best_fit = {
                "fit_parameters": fit_parameters,
                "parameters_cov": parameters_cov,
                "sum_sq_residuals": sum_sq_residuals,
            }

    if best_fit is None:
        raise RuntimeError("fit failed")

    # The output results are stored in a json, but ndarray is not JSON serializable,
    # so the parameters are converted to list.
    parameter_errors = np.sqrt(np.diag(best_fit["parameters_cov"])).tolist()
    model_parameters = best_fit["fit_parameters"].tolist()
    return (
        model_parameters[1],
        model_parameters,
        parameter_errors,
    )


def _fit(data: QubitSpectroscopyData) -> QubitSpectroscopyResults:
    """Post-processing function for QubitSpectroscopy."""
    qubits = data.qubits
    frequency = {}
    fitted_parameters = {}
    for qubit in qubits:
        fit_result = _lorentzian_fit(data[qubit])
        if fit_result is not None:
            frequency[qubit], fitted_parameters[qubit], _ = fit_result
    return QubitSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        amplitude=data.amplitudes,
    )


def _plot(data: QubitSpectroscopyData, target: QubitId, fit: QubitSpectroscopyResults):
    figures = []
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    qubit_data = data[target]
    fitting_report = ""
    frequencies = qubit_data.freq
    signal = qubit_data.signal

    phase = qubit_data.phase
    fig.add_trace(
        go.Scatter(
            x=frequencies * scipy.constants.nano,
            y=signal,
            opacity=1,
            name="Frequency",
            showlegend=True,
            legendgroup="Frequency",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=frequencies * scipy.constants.nano,
            y=phase,
            opacity=1,
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
        ),
        row=1,
        col=2,
    )

    freqrange = np.linspace(
        min(frequencies),
        max(frequencies),
        2 * len(frequencies),
    )

    if fit is not None:
        params = fit.fitted_parameters[target]
        fig.add_trace(
            go.Scatter(
                x=freqrange * scipy.constants.nano,
                y=_lorentzian_with_offset(freqrange, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        if data.amplitudes[target] is not None:
            labels = ["Qubit Frequency [Hz]", "Drive Amplitude [a.u.]"]
            values = [fit.frequency[target], data.amplitudes[target]]

            fitting_report = table_html(
                table_dict(
                    target,
                    labels,
                    values,
                )
            )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHz]",
        yaxis_title="Signal [a.u.]",
        xaxis2_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
    )
    figures.append(fig)

    return figures, fitting_report


def _update(
    results: QubitSpectroscopyResults, platform: CalibrationPlatform, target: QubitId
):
    platform.calibration.single_qubits[target].qubit.frequency_01 = results.frequency[
        target
    ]
    update.drive_frequency(results.frequency[target], platform, target)


qubit_spectroscopy = Protocol(_acquisition, _fit, _plot, _update)
"""Qubit Spectroscopy routine.
"""
