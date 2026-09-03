"""Cryoscope experiment."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.linalg
import scipy.optimize
import scipy.signal
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    BaseEnvelope,
    Delay,
    ExponentialFilter,
    Parameter,
    Platform,
    Pulse,
    PulseId,
    PulseSequence,
    Sweeper,
    Waveform,
)

from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.config import log
from qibocal.protocols.ramsey.processing import fitting
from qibocal.protocols.utils import table_dict, table_html

__all__ = ["CryoscopeData", "CryoscopeResults", "cryoscope"]


NYQUIST_CYCLES_PER_SAMPLE = 0.5
"""Cycles per sample at the Nyquist limit

Nyquist frequency = NYQUIST_CYCLES_PER_SAMPLE * sampling rate
"""

BUFFER_TIME = 100
"""Extra time in ns between the two pi/2 pulses.

Set to 100 ns following the Cryoscope paper https://arxiv.org/abs/1907.04818
"""
# TODO: According to PycQED this "needs some playing around sometimes", so perhaps
# should be exposed to the user as an input parameter. See
# https://github.com/DiCarloLab-Delft/PycQED_py3/blob/dcf05e699608ea434ddd727fe538ce7cfd9ece68/pycqed/analysis/tools/cryoscope_tools.py#L97

DERIVATIVE_WINDOW_SIZE = 7
"""Size, in samples, of the Savitzky-Golay window used for the derivative."""


def to_samples(duration: float, sampling_rate: float) -> int:
    """Convert a duration in ns to the number of samples.

    Raises an error if the duration does not correspond to an integer number of samples
    at the given sampling rate [GSps].
    """
    exact = duration * sampling_rate
    rounded = round(exact)
    if not np.isclose(exact, rounded):
        raise ValueError(
            f"A duration of {duration} ns is not a whole number of samples at a "
            f"sampling rate of {sampling_rate} GSps: it corresponds to {exact} samples."
        )
    return rounded


class PaddedRectangular(BaseEnvelope):
    """Rectangular envelope with a fixed number of leading zero samples.

    The waveform consists of ``padding_samples`` zeros followed by unit-amplitude
    samples. This allows short flux pulses to be represented at the waveform level, even
    when the hardware's pulse scheduling granularity is coarser than the desired pulse
    duration.
    """

    kind: Literal["padded_rectangular"] = "padded_rectangular"
    padding_samples: int
    """Number of leading zero samples."""

    def i(self, samples: int) -> Waveform:
        """Return a rectangular envelope with `padding_samples` leading zeros."""
        if samples < self.padding_samples:
            raise ValueError(
                f"samples ({samples}) must be >= padding ({self.padding_samples})"
            )
        return np.concatenate(
            [np.zeros(self.padding_samples), np.ones(samples - self.padding_samples)]
        )


@dataclass
class CryoscopeParameters(Parameters):
    """Cryoscope user inputs."""

    duration_max: float
    """Maximum flux pulse duration [ns]."""
    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    fir: int
    """Number of feedforward taps to be optimized after IIR."""
    iir: bool
    """Whether an IIR filter should be determined.
    If False only an FIR filter is determined.
    """
    padding_duration: float = 0
    """Duration in ns of the leading zeros in the flux pulse.

    Padding is fixed during the duration sweep and added before the pulse. The waveform
    consists of `padding_duration` ns of zeros followed by `duration` ns of rectangular
    samples, for a total length of `padding_duration + duration`.

    Useful when hardware enforces a minimum pulse length.
    """


@dataclass
class CryoscopeResults(Results):
    """Cryoscope outputs."""

    fitted_parameters: dict[tuple[QubitId, str], list[float]] = field(
        default_factory=dict
    )
    """Fitted <X> and <Y> for each qubit."""
    detuning: dict[QubitId, list[float]] = field(default_factory=dict)
    """Expected detuning."""
    amplitude: dict[QubitId, list[float]] = field(default_factory=dict)
    """Flux amplitude computed from detuning."""
    step_response: dict[QubitId, list[float]] = field(default_factory=dict)
    """Waveform normalized to 1."""
    exp_amplitude: dict[QubitId, float] = field(default_factory=dict)
    """A parameters for the exp decay approximation"""
    tau: dict[QubitId, float] = field(default_factory=dict)
    """Time decay constant in exp decay approximation [ns]."""
    fir_taps: dict[QubitId, list[float]] = field(default_factory=dict)
    """FIR feedforward taps"""

    # TODO: this is here for the plotting for now, but needs to go
    feedforward_taps: dict[QubitId, list[float]] = field(default_factory=dict)
    """feedforward taps"""
    feedforward_taps_iir: dict[QubitId, list[float]] = field(default_factory=dict)
    """feedforward taps for IIR"""
    feedback_taps: dict[QubitId, list[float]] = field(default_factory=dict)
    """feedback taps"""

    # TODO: we only need this because params is not passed to the fit function
    iir: bool = False
    """Whether an IIR filter should be determined.
    If False only an FIR filter is determined.
    """

    def __contains__(self, key):
        return key in self.detuning


@dataclass(frozen=True)
class XYSequences:
    """Sequences to be executed for a single qubit."""

    sequences: dict[str, PulseSequence]
    """Sequences for measuring both the X and Y coordinates."""
    readout_ids: dict[str, PulseId]
    """Id of the readout pulse of each sequences, to retrieve its results."""
    flux_pulse: Pulse
    """Flux pulse shared by the two sequences."""


def generate_sequences(
    platform: Platform,
    qubit: QubitId,
    params: CryoscopeParameters,
) -> XYSequences:
    """Compute sequences for <X> and <Y> with a flux pulse ready for duration sweep."""
    native = platform.natives.single_qubit[qubit]

    drive_channel, ry90 = native.R(theta=np.pi / 2, phi=np.pi / 2)[0]
    _, rx90 = native.R(theta=np.pi / 2)[0]
    ro_channel, ro_pulse_x = native.MZ()[0]
    ro_pulse_y = (
        ro_pulse_x.new()
    )  # To ensure X and Y ro pulses don't have the same UUID
    flux_channel = platform.qubits[qubit].flux
    assert flux_channel is not None

    # model_construct skips validation because PaddedRectangular is not a supported
    # qibolab envelope. The pulse is not serialized, so this is acceptable here.
    assert platform.sampling_rate is not None
    flux_pulse = Pulse.model_construct(
        duration=params.padding_duration,
        amplitude=params.flux_pulse_amplitude,
        envelope=PaddedRectangular(
            padding_samples=to_samples(params.padding_duration, platform.sampling_rate)
        ),
    )

    # the two pi/2 pulses are separated by a fixed separation time
    separation_time = params.duration_max + params.padding_duration + BUFFER_TIME

    # create the sequences
    sequence_x = PulseSequence(
        [
            (drive_channel, ry90),
            (flux_channel, Delay(duration=ry90.duration)),
            (flux_channel, flux_pulse),
            (drive_channel, Delay(duration=separation_time)),
            (drive_channel, ry90),
            (
                ro_channel,
                Delay(duration=ry90.duration + separation_time + ry90.duration),
            ),
            (ro_channel, ro_pulse_x),
        ]
    )

    sequence_y = PulseSequence(
        [
            (drive_channel, ry90),
            (flux_channel, Delay(duration=rx90.duration)),
            (flux_channel, flux_pulse),
            (drive_channel, Delay(duration=separation_time)),
            (drive_channel, rx90),
            (
                ro_channel,
                Delay(duration=ry90.duration + separation_time + rx90.duration),
            ),
            (ro_channel, ro_pulse_y),
        ]
    )
    return XYSequences(
        sequences={"MX": sequence_x, "MY": sequence_y},
        readout_ids={"MX": ro_pulse_x.id, "MY": ro_pulse_y.id},
        flux_pulse=flux_pulse,
    )


@dataclass
class CryoscopeData(Data):
    """Cryoscope acquisition outputs."""

    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    fir: int
    """Number of feedforward taps to be optimized after IIR."""
    sampling_rate: float
    """Sampling rate of the instrument [GSps]."""
    flux_pulse_durations: list[float]
    """Durations of the flux pulses [ns]. Same for all qubits."""
    flux_coefficients: dict[QubitId, list[float]] = field(default_factory=dict)
    """Flux - amplitude relation coefficients obtained from flux_amplitude_frequency routine"""
    has_filters: dict[QubitId, bool] = field(default_factory=dict)
    """Check if there are filters already."""
    data: dict[tuple[QubitId, str], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    # TODO: we only need this because params is not passed to the fit function
    iir: bool = False
    """Whether an IIR filter should be determined.
    If False only an FIR filter is determined.
    """


def _check_phase_can_be_unwrapped(
    flux_coefficients: list[float],
    flux_pulse_amplitude: float,
    sampling_rate: float,
    qubit: QubitId,
) -> None:
    """Check if the sampling rate is above the Nyquist rate."""
    f = np.poly1d(flux_coefficients)
    detuning = abs(f(flux_pulse_amplitude) - f(0.0))  # GHz
    cycles_per_sample = detuning / sampling_rate
    if cycles_per_sample > NYQUIST_CYCLES_PER_SAMPLE:
        raise ValueError(
            f"Cannot unwrap the phase for qubit {qubit}: the expected detuning is "
            f"{detuning:.3f} GHz, resulting in {cycles_per_sample:.3f} cycles per "
            f"sample ({1 / sampling_rate} ns). This is above the Nyquist limit. "
            "Reduce flux_pulse_amplitude."
        )


def _acquisition(
    params: CryoscopeParameters,
    platform: Platform,
    targets: list[QubitId],
) -> CryoscopeData:
    """Acquisition for cryoscope experiment.

    The following sequence is played for each qubit.

    drive    --- RY90 ------------------- RY90 -------
    flux     --------- FluxPulse(t) ------------------
    readout  ----------------------------------- MZ --

    The previous sequence measures <X>, to measure <Y> the second drive pulse
    is replaced with RX90.
    The delay between the two pi/2 pulses is fixed at the maximum length of the flux
    pulse (padding included) + 100 ns (following the paper).
    """
    sampling_rate = platform.sampling_rate
    assert sampling_rate is not None
    # the duration is swept one sample at a time, starting from the first sample after
    # the step edge, since the filters are defined on consecutive samples counted from
    # the beginning of the pulse
    durations = (
        np.arange(1, to_samples(params.duration_max, sampling_rate)) / sampling_rate
    )

    iir_free_parameters = params.iir * 2
    if params.fir + iir_free_parameters > len(durations):
        raise ValueError(
            f"Cannot fit {params.fir} FIR taps and {iir_free_parameters} exponential "
            f"parameters with only {len(durations)} duration points: the fit would be "
            "underdetermined."
        )

    data = CryoscopeData(
        fir=params.fir,
        flux_pulse_amplitude=params.flux_pulse_amplitude,
        iir=params.iir,
        sampling_rate=sampling_rate,
        flux_pulse_durations=durations.tolist(),
    )

    for qubit in targets:
        if platform.calibration.single_qubits[qubit].qubit.flux_coefficients is None:
            raise ValueError(
                "Cannot run cryoscope without flux coefficients, run "
                f"cryoscope amplitude on qubit {qubit} before the cryoscope"
            )

        data.flux_coefficients[qubit] = platform.calibration.single_qubits[
            qubit
        ].qubit.flux_coefficients
        data.has_filters[qubit] = bool(
            platform.config(platform.qubits[qubit].flux).filters
        )
        _check_phase_can_be_unwrapped(
            data.flux_coefficients[qubit],
            data.flux_pulse_amplitude,
            data.sampling_rate,
            qubit,
        )

    qubit_to_xy_sequences = {
        qubit: generate_sequences(platform, qubit, params) for qubit in targets
    }

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=durations + params.padding_duration,
        pulses=[qs.flux_pulse for qs in qubit_to_xy_sequences.values()],
    )

    options = {
        "nshots": params.nshots,
        "acquisition_type": AcquisitionType.DISCRIMINATION,
        "averaging_mode": AveragingMode.CYCLIC,
    }

    results = platform.execute(
        [
            sum(
                (qs.sequences[meas] for qs in qubit_to_xy_sequences.values()),
                PulseSequence(),
            )
            for meas in ["MX", "MY"]
        ],
        [[sweeper]],
        **options,
    )

    for qubit, qs in qubit_to_xy_sequences.items():
        for measure, readout_id in qs.readout_ids.items():
            data.data[qubit, measure] = results[readout_id]

    return data


def exponential_params(
    step_response: npt.ArrayLike, durations: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Fit an exponential distortion."""

    def _expmodel(t, tau, exp_amplitude, g):
        return g * (1 + exp_amplitude * np.exp(-t / tau))

    popt, _ = scipy.optimize.curve_fit(
        _expmodel,
        durations,
        step_response,
    )
    return popt


# TODO: refactor into sub-functions with smaller scopes
def _fit(data: CryoscopeData) -> CryoscopeResults:
    """Postprocessing for cryoscope experiment.

    From <X> and <Y> we compute the expecting step response.
    The complex data <X> + i <Y> are demodulated using the frequency found
    by fitting a sinusoid to both <X> and <Y>.
    Next, the phase is computed and finally the detuning using a savgol_filter.
    The "real" detuning is computed by reintroducing the demodulation frequency.
    Finally, using the parameters given by the flux_amplitude_frequency experiment,
    we compute the expected flux_amplitude by inverting the formula:

    f = c_1 A^2 + c_2 A + c_3

    where f is the detuning and A is the flux amplitude.
    The step response is computed by normalizing the amplitude by its value computed above.
    For some of the manipulations see: https://github.com/DiCarloLab-Delft/PycQED_py3/blob/c4279cbebd97748dc47127e56f6225021f169257/pycqed/analysis/tools/cryoscope_tools.py#L73
    """

    # The check in _check_phase_can_be_unwrapped ensures that nyquist_order = 0 is
    # always true.
    nyquist_order = 0

    fitted_parameters = {}
    detuning = {}
    amplitude = {}
    step_response = {}
    alpha = {}
    g = {}
    time_decay = {}
    feedforward_taps_iir = {}
    feedforward_taps = {}
    fir_taps = {}
    feedback_taps = {}
    durations = np.array(data.flux_pulse_durations)
    for qubit, setup in data.data:
        y = 1 - 2 * data[qubit, setup]
        popt, _ = fitting(durations, y)

        fitted_parameters[qubit, setup] = popt

    qubits = np.unique([i[0] for i in data.data]).tolist()

    sampling_rate = data.sampling_rate
    for qubit in qubits:
        X_exp = 2 * data[(qubit, "MX")] - 1
        Y_exp = 1 - 2 * data[(qubit, "MY")]

        norm_data = X_exp + 1j * Y_exp

        # demodulation frequency in GHz found by fitting sinusoidal
        demod_freq = -fitted_parameters[qubit, "MX"][2] / 2 / np.pi
        # to be used in savgol_filter
        derivative_window_size = max(3, DERIVATIVE_WINDOW_SIZE)
        derivative_window_size += (derivative_window_size + 1) % 2

        # find demodulatation frequency
        demod_data = np.exp(2 * np.pi * 1j * durations * np.abs(demod_freq)) * (
            norm_data
        )

        # compute phase
        phase = np.unwrap(np.angle(demod_data))
        phase -= phase[0]
        # compute detuning in GHz
        raw_detuning = (
            scipy.signal.savgol_filter(
                phase / (2 * np.pi),
                window_length=derivative_window_size,
                polyorder=2,
                deriv=1,
            )
            * sampling_rate
        )
        detuning[qubit] = (
            raw_detuning + demod_freq + sampling_rate * nyquist_order
        ).tolist()

        # invert frequency amplitude formula
        p = np.poly1d(data.flux_coefficients[qubit])
        amplitude[qubit] = [max((p - freq).roots).real for freq in detuning[qubit]]

        step_response[qubit] = (
            np.array(amplitude[qubit]) / data.flux_pulse_amplitude
        ).tolist()
        if not data.has_filters[qubit]:
            # Derive IIR
            if data.iir:
                exp_params = exponential_params(step_response[qubit], durations)
                tau, exp_amplitude, _ = exp_params
                iir_filter = ExponentialFilter(
                    amplitude=exp_amplitude, tau=round(tau * sampling_rate)
                )
                feedback_taps[qubit] = iir_filter.feedback
                feedforward_taps_iir[qubit] = iir_filter.feedforward
            else:
                exp_params = [0.0, 0.0, 1.0]
                feedback_taps[qubit] = [1.0]
                feedforward_taps_iir[qubit] = [1.0]

            time_decay[qubit], alpha[qubit], g[qubit] = exp_params
            iir_correction = scipy.signal.lfilter(
                feedforward_taps_iir[qubit], feedback_taps[qubit], step_response[qubit]
            )
            # FIR corrections

            taps = data.fir
            baseline = g[qubit]

            # The Toeplitz matrix is lower triangular, with zeros in the upper triangle.
            # Its diagonals contain successive shifts of iir_correction, so multiplying
            # by the matrix implements the discrete convolution with the FIR taps.
            toeplitz_matrix = scipy.linalg.toeplitz(iir_correction, np.zeros(taps))
            # solve: toeplitz_matrix @ fir == baseline
            fir, _, _, _ = np.linalg.lstsq(
                toeplitz_matrix, np.full(len(iir_correction), baseline)
            )
            fir_taps[qubit] = fir.tolist()
            feedforward_taps[qubit] = np.convolve(
                feedforward_taps_iir[qubit], fir
            ).tolist()

    return CryoscopeResults(
        amplitude=amplitude,
        detuning=detuning,
        step_response=step_response,
        fitted_parameters=fitted_parameters,
        exp_amplitude=alpha,
        tau=time_decay,
        feedforward_taps=feedforward_taps,
        feedforward_taps_iir=feedforward_taps_iir,
        feedback_taps=feedback_taps,
        fir_taps=fir_taps,
        iir=data.iir,
    )


def _plot(data: CryoscopeData, fit: CryoscopeResults, target: QubitId):
    """Cryoscope plots."""

    fig = make_subplots(
        rows=2,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )
    duration = data.flux_pulse_durations

    fig.add_trace(
        go.Scatter(
            x=duration,
            y=2 * data[(target, "MX")] - 1,
            name="X",
            legendgroup="1",
            mode="markers",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=duration,
            y=1 - 2 * data[(target, "MY")],
            name="Y",
            legendgroup="1",
            mode="markers",
        ),
        row=1,
        col=1,
    )

    fitting_report = ""
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=duration,
                y=fit.step_response[target],
                name="Uncorrected waveform",
                legendgroup="2",
                mode="lines",
            ),
            row=2,
            col=1,
        )

        if not data.has_filters[target]:
            all_corrections = scipy.signal.lfilter(
                fit.feedforward_taps[target],
                fit.feedback_taps[target],
                fit.step_response[target],
            )

            if data.iir:
                iir_corrections = scipy.signal.lfilter(
                    fit.feedforward_taps_iir[target],
                    fit.feedback_taps[target],
                    fit.step_response[target],
                )
                fig.add_trace(
                    go.Scatter(
                        x=duration,
                        y=iir_corrections,
                        name="IIR corrections",
                        legendgroup="2",
                        mode="lines",
                    ),
                    row=2,
                    col=1,
                )

            fig.add_trace(
                go.Scatter(
                    x=duration,
                    y=all_corrections,
                    name="FIR + IIR corrections" if data.iir else "FIR corrections",
                    legendgroup="2",
                    mode="lines",
                ),
                row=2,
                col=1,
            )

            exp_amplitude = fit.exp_amplitude[target]
            tau = fit.tau[target]

            fitting_report = table_html(
                table_dict(
                    target,
                    ["A", "tau [ns]"],
                    [
                        exp_amplitude,
                        tau,
                    ],
                )
            )

        fig.update_layout(
            showlegend=True,
            legend_tracegroupgap=120,
            xaxis2_title="Duration [ns]",
            yaxis1_title="Expectation value",
            yaxis2_title="Waveform",
        )

    return [fig], fitting_report


def _update(results: CryoscopeResults, platform: Platform, target: QubitId):
    if platform.config(platform.qubits[target].flux).filters:
        log.info(
            f"Qubit {target} already has filters on its flux channel, "
            "skipping the filters update."
        )
        return

    filters = [{"kind": "fir", "coefficients": results.fir_taps[target]}]
    # TODO: multiple iir filters?
    if results.iir:
        assert platform.sampling_rate is not None
        filters.append(
            {
                "kind": "exp",
                "amplitude": results.exp_amplitude[target],
                "tau": results.tau[target] * platform.sampling_rate,
            }
        )
    platform.update({f"configs.{platform.qubits[target].flux}.filters": filters})


cryoscope = Protocol(_acquisition, _fit, _plot, _update)
