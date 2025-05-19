"""Cryoscope experiment."""

from dataclasses import dataclass, field

import cma
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy
import scipy.signal
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Platform,
    Pulse,
    PulseSequence,
    Rectangular,
)
from scipy.optimize import curve_fit
from scipy.signal import lfilter

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.config import log
from qibocal.protocols.ramsey.utils import fitting
from qibocal.protocols.utils import table_dict, table_html

# TODO: remove hard-coded QM parameters
FEEDFORWARD_MAX = 2 - 2**-16
"""Maximum feedforward tap value"""
FEEDBACK_MAX = 1 - 2**-20
"""Maximum feedback tap value"""

__all__ = ["cryoscope", "CryoscopeData", "CryoscopeResults"]


@dataclass
class CryoscopeParameters(Parameters):
    """Cryoscope runcard inputs."""

    duration_min: float
    """Minimum flux pulse duration."""
    duration_max: float
    """Maximum flux duration start."""
    duration_step: float
    """Flux pulse duration step."""
    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    fir: int = 20
    """Number of feedforward taps to be optimized after IIR."""
    unrolling: bool = True


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
    exp_amplitude: dict[QubitId, list[float]] = field(default_factory=dict)
    """A parameters for the exp decay approximation"""
    tau: dict[QubitId, list[float]] = field(default_factory=dict)
    """time decay constant in exp decay approximation"""
    feedforward_taps: dict[QubitId, list[float]] = field(default_factory=dict)
    """feedforward taps"""
    feedforward_taps_iir: dict[QubitId, list[float]] = field(default_factory=dict)
    """feedforward taps for IIR"""
    feedback_taps: dict[QubitId, list[float]] = field(default_factory=dict)
    """feedback taps"""

    def __contains__(self, key):
        return key in self.detuning


CryoscopeType = np.dtype([("duration", float), ("prob_1", np.float64)])
"""Custom dtype for Cryoscope."""


def generate_sequences(
    platform: Platform,
    qubit: QubitId,
    duration: float,
    params: CryoscopeParameters,
) -> tuple[PulseSequence, PulseSequence]:
    """Compute sequences at fixed duration of flux pulse for <X> and <Y>"""
    native = platform.natives.single_qubit[qubit]

    drive_channel, ry90 = native.R(theta=np.pi / 2, phi=np.pi / 2)[0]
    _, rx90 = native.R(theta=np.pi / 2)[0]
    ro_channel, ro_pulse = native.MZ()[0]
    flux_channel = platform.qubits[qubit].flux

    flux_pulse = Pulse(
        duration=duration,
        amplitude=params.flux_pulse_amplitude,
        envelope=Rectangular(),
    )

    # create the sequences
    sequence_x, sequence_y = PulseSequence(), PulseSequence()

    sequence_x.extend(
        [
            (drive_channel, ry90),
            (flux_channel, Delay(duration=ry90.duration)),
            (flux_channel, flux_pulse),
            (drive_channel, Delay(duration=params.duration_max + 100)),
            (drive_channel, ry90),
            (
                ro_channel,
                Delay(
                    duration=ry90.duration + params.duration_max + 100 + ry90.duration
                ),
            ),
            (ro_channel, ro_pulse),
        ]
    )

    sequence_y.extend(
        [
            (drive_channel, ry90),
            (flux_channel, Delay(duration=rx90.duration)),
            (flux_channel, flux_pulse),
            (drive_channel, Delay(duration=params.duration_max + 100)),
            (drive_channel, rx90),
            (
                ro_channel,
                Delay(
                    duration=ry90.duration + params.duration_max + 100 + rx90.duration
                ),
            ),
            (ro_channel, ro_pulse),
        ]
    )
    return sequence_x, sequence_y


@dataclass
class CryoscopeData(Data):
    """Cryoscope acquisition outputs."""

    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    fir: int
    """Number of feedforward taps to be optimized after IIR."""
    flux_coefficients: dict[QubitId, list[float]] = field(default_factory=dict)
    """Flux - amplitude relation coefficients obtained from flux_amplitude_frequency routine"""
    filters: dict[QubitId, float] = field(default_factory=dict)
    """Check if there are filters already."""
    data: dict[tuple[QubitId, str], npt.NDArray[CryoscopeType]] = field(
        default_factory=dict
    )

    def has_filters(self, qubit: QubitId) -> bool:
        """Checking if for qubit there are already filters."""
        try:
            return len(self.filters[qubit]) > 0
        except AttributeError:
            return False


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
    The delay between the two pi/2 pulses is fixed at t_max (maximum duration of flux pulse)
    + 100 ns (following the paper).
    """
    data = CryoscopeData(
        fir=params.fir,
        flux_pulse_amplitude=params.flux_pulse_amplitude,
    )

    for qubit in targets:
        assert (
            platform.calibration.single_qubits[qubit].qubit.flux_coefficients
            is not None
        ), (
            f"Cannot run cryoscope without flux coefficients, run cryoscope amplitude on qubit {qubit} before the cryoscope"
        )

        data.flux_coefficients[qubit] = platform.calibration.single_qubits[
            qubit
        ].qubit.flux_coefficients
        data.filters[qubit] = platform.config(platform.qubits[qubit].flux).filter

    sequences_x = []
    sequences_y = []

    duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    for duration in duration_range:
        sequence_x = PulseSequence()
        sequence_y = PulseSequence()

        for qubit in targets:
            qubit_sequence_x, qubit_sequence_y = generate_sequences(
                platform, qubit, duration, params
            )
            sequence_x += qubit_sequence_x
            sequence_y += qubit_sequence_y

        sequences_x.append(sequence_x)
        sequences_y.append(sequence_y)

    options = dict(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    if params.unrolling:
        results_x = platform.execute(sequences_x, **options)
        results_y = platform.execute(sequences_y, **options)
    else:
        results_x = [
            platform.execute([sequence], **options) for sequence in sequences_x
        ]
        results_y = [
            platform.execute([sequence], **options) for sequence in sequences_y
        ]

    for measure, results, sequence in zip(
        ["MX", "MY"], [results_x, results_y], [sequences_x, sequences_y]
    ):
        for i, (duration, sequence) in enumerate(zip(duration_range, sequence)):
            for qubit in targets:
                ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[
                    -1
                ]
                result = (
                    results[ro_pulse.id]
                    if params.unrolling
                    else results[i][ro_pulse.id]
                )
                data.register_qubit(
                    CryoscopeType,
                    (qubit, measure),
                    dict(
                        duration=np.array([duration]),
                        prob_1=result,
                    ),
                )

    return data


def exponential_params(step_response, acquisition_time):
    t = np.arange(0, acquisition_time, 1)
    init_guess = [1, 10, 1]
    target = np.ones(len(t))

    def expmodel(t, tau, exp_amplitude, g):
        return step_response / (g * (1 + exp_amplitude * np.exp(-t / tau)))

    popt, _ = curve_fit(expmodel, t, target, p0=init_guess)

    return popt


def filter_calc(params, sampling_rate):
    tau, exp_amplitude, _ = params
    alpha = 1 - np.exp(-1 / (sampling_rate * tau * (1 + exp_amplitude)))
    k = (
        exp_amplitude / ((1 + exp_amplitude) * (1 - alpha))
        if exp_amplitude < 0
        else exp_amplitude / (1 + exp_amplitude - alpha)
    )
    b0 = 1 - k + k * alpha
    b1 = -(1 - k) * (1 - alpha)
    a0 = 1
    a1 = -(1 - alpha)

    feedback_taps = np.array([a0, a1])
    feedforward_taps = np.array([b0, b1])

    if np.any(np.abs(feedback_taps) > FEEDBACK_MAX):
        feedback_taps[feedback_taps > FEEDBACK_MAX] = FEEDBACK_MAX
        feedback_taps[feedback_taps < -FEEDBACK_MAX] = -FEEDBACK_MAX

    return feedback_taps.tolist(), feedforward_taps.tolist()


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
    feedback_taps = {}
    for qubit, setup in data.data:
        qubit_data = data[qubit, setup]
        x = qubit_data.duration
        y = 1 - 2 * qubit_data.prob_1

        popt, _ = fitting(x, y)

        fitted_parameters[qubit, setup] = popt

    qubits = np.unique([i[0] for i in data.data]).tolist()

    for qubit in qubits:
        sampling_rate = 1 / (x[1] - x[0])
        X_exp = 2 * data[(qubit, "MX")].prob_1 - 1
        Y_exp = 1 - 2 * data[(qubit, "MY")].prob_1

        norm_data = X_exp + 1j * Y_exp

        # demodulation frequency found by fitting sinusoidal
        demod_freq = -fitted_parameters[qubit, "MX"][2] / 2 / np.pi * sampling_rate
        # to be used in savgol_filter
        derivative_window_length = 7 / sampling_rate
        derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
        derivative_window_size += (derivative_window_size + 1) % 2

        # find demodulatation frequency
        demod_data = np.exp(2 * np.pi * 1j * x * np.abs(demod_freq)) * (norm_data)

        # compute phase
        phase = np.unwrap(np.angle(demod_data))
        phase -= phase[0]
        # compute detuning
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
        if not data.has_filters(qubit):
            # Derive IIR
            acquisition_time = len(x)
            exp_params = exponential_params(step_response[qubit], acquisition_time)
            feedback_taps[qubit], feedforward_taps_iir[qubit] = filter_calc(
                exp_params, sampling_rate
            )
            time_decay[qubit], alpha[qubit], g[qubit] = exp_params
            iir_correction = lfilter(
                feedforward_taps_iir[qubit], feedback_taps[qubit], step_response[qubit]
            )
            # FIR corrections

            taps = data.fir
            baseline = g[qubit]
            x0 = [1] + (taps - 1) * [0]

            def fir_cost_function(x):
                yc = lfilter(x, 1, iir_correction)
                return np.mean(np.abs(yc - baseline)) / np.abs(baseline)

            fir = cma.fmin2(fir_cost_function, x0, 0.5)[0]

            feedforward_taps[qubit] = np.convolve(
                feedforward_taps_iir[qubit], fir
            ).tolist()

            if np.max(np.abs(feedforward_taps[qubit])) > FEEDFORWARD_MAX:
                feedforward_taps[qubit] = (
                    2
                    * np.array(feedforward_taps[qubit])
                    / abs(max(feedforward_taps[qubit]))
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
    )


def _plot(data: CryoscopeData, fit: CryoscopeResults, target: QubitId):
    """Cryoscope plots."""

    fig = make_subplots(
        rows=2,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )
    duration = data[(target, "MX")].duration

    fig.add_trace(
        go.Scatter(
            x=duration,
            y=2 * data[(target, "MX")].prob_1 - 1,
            name="X",
            legendgroup="1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=duration,
            y=1 - 2 * data[(target, "MY")].prob_1,
            name="Y",
            legendgroup="1",
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
            ),
            row=2,
            col=1,
        )

        if not data.has_filters(target):
            iir_corrections = lfilter(
                fit.feedforward_taps_iir[target],
                fit.feedback_taps[target],
                fit.step_response[target],
            )
            all_corrections = lfilter(
                fit.feedforward_taps[target],
                fit.feedback_taps[target],
                fit.step_response[target],
            )

            fig.add_trace(
                go.Scatter(
                    x=duration,
                    y=iir_corrections,
                    name="IIR corrections",
                    legendgroup="2",
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=duration,
                    y=all_corrections,
                    name="FIR + IIR corrections",
                    legendgroup="2",
                ),
                row=2,
                col=1,
            )

            exp_amplitude = fit.exp_amplitude[target]
            tau = fit.tau[target]

            fitting_report = table_html(
                table_dict(
                    target,
                    ["A", "tau"],
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
    try:
        update.feedforward(results.feedforward_taps[target], platform, target)
        update.feedback(results.feedback_taps[target], platform, target)
    except KeyError:
        log.info(f"Skipping filters update on qubit {target}.")


cryoscope = Routine(_acquisition, _fit, _plot, _update)
