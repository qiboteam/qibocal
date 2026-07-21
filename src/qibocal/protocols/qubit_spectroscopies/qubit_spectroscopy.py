from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.constants
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    ChannelId,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)
from qibolab._core.components import IqChannel
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    Range,
    RangeLike,
    lorentzian,
    table_dict,
    table_html,
    to_range,
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

    frequency: RangeLike | None = None
    """Drive frequency [Hz] range for sweep."""
    freq_width: int | None = None
    """Width [Hz] for frequency sweep relative  to the qubit frequency."""
    freq_step: int | None = None
    """Frequency [Hz] step for sweep."""
    drive_duration: int = 4000
    """Drive pulse duration [ns]. Same for all qubits."""
    drive_amplitude: float = 1.0
    """Drive pulse amplitude (optional). Same for all qubits."""

    def frequency_range(self, center: float = 0.0) -> Range:
        def legacy_range() -> Range:
            assert self.freq_width is not None and self.freq_step is not None
            return (
                center - self.freq_width / 2,
                center + self.freq_width / 2,
                self.freq_step,
            )

        return (
            to_range(self.frequency, center=center)
            if self.frequency is not None
            else legacy_range()
        )


@dataclass
class QubitSpectroscopyResults(Results):
    """QubitSpectroscopy outputs."""

    frequency: dict[QubitId, float]
    """Drive frequecy [GHz] for each qubit."""
    amplitude: dict[QubitId, float]
    """Input drive amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""


@dataclass
class QubitSpectroscopyData(Data):
    """QubitSpectroscopy acquisition outputs."""

    drive_frequencies: dict[QubitId, list[float]]
    """Frequencies."""
    amplitudes: dict[QubitId, float]
    """Amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray[np.float64]]
    """Raw data acquired."""

    def signal(self, qubit: QubitId) -> npt.NDArray[np.float64]:
        return magnitude(self.data[qubit])

    def phase(self, qubit: QubitId) -> npt.NDArray[np.float64]:
        return phase(self.data[qubit])


def _calculate_batches(
    freq_width: float, max_if_bandwidth: float = 300e6
) -> npt.NDArray[np.float64]:
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
    freq_range = params.frequency_range()
    width = freq_range[1] - freq_range[0]
    batches = _calculate_batches(freq_width=width)

    # Get drive channels and LO channels for each qubit
    drive_channels: dict[QubitId, ChannelId] = {}
    lo_channels: dict[QubitId, str | None] = {}
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        drive_channels[qubit] = platform.qubits[qubit].drive

        # Get the LO channel associated with this drive channel
        channel_obj = platform.channels[drive_channels[qubit]]
        if isinstance(channel_obj, IqChannel) and channel_obj.lo is not None:
            lo_channels[qubit] = channel_obj.lo
        else:
            lo_channels[qubit] = None

    # Initialize storage for intermediate results
    values = {qubit: defaultdict(list) for qubit in targets}
    # TODO: remove, and propagate differently, since it is always the same for
    # all qubits
    drive_amplitudes: dict[QubitId, float] = {
        q: params.drive_amplitude for q in targets
    }

    # Execute each batch
    for start, end, lo_offset in batches:
        delta_frequency_range = np.arange(start, end, freq_range[2])

        # Build the pulse sequence
        sequence = PulseSequence()
        ro_pulses = {}
        sweepers = []

        for qubit in targets:
            natives = platform.natives.single_qubit[qubit]
            qd_channel = drive_channels[qubit]
            assert natives.MZ is not None
            ro_channel, ro_pulse = natives.MZ()[0]

            qd_pulse = Pulse(
                amplitude=params.drive_amplitude,
                duration=params.drive_duration,
                envelope=Rectangular(),
            )
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

            # Store results with absolute frequencies
            values[qubit]["frequency"].append(delta_frequency_range + f0)
            values[qubit]["result"].append(result)

    freqs: dict[QubitId, list[float]] = {}
    data: dict[QubitId, npt.NDArray[np.float64]] = {}
    # Combine all batches for each qubit
    for q in targets:
        # Concatenate arrays from all batches
        freqs[q] = np.concatenate(values[q]["frequency"]).tolist()
        data[q] = np.concatenate(values[q]["result"])

    return QubitSpectroscopyData(
        drive_frequencies=freqs, amplitudes=drive_amplitudes, data=data
    )


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


def _lorentzian_fit(
    frequencies: np.ndarray, signal: np.ndarray
) -> tuple[float, list[float], list[float]]:
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
    frequency: dict[QubitId, float] = {}
    params: dict[QubitId, list[float]] = {}
    for qubit in qubits:
        fit_result = _lorentzian_fit(
            np.array(data.drive_frequencies[qubit]), magnitude(data.data[qubit])
        )
        frequency[qubit], params[qubit], _ = fit_result
    return QubitSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=params,
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
    fitting_report = ""
    frequencies = np.array(data.drive_frequencies[target])
    signal = data.signal(target)
    phase = data.phase(target)

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
