"""Cryoscope experiment."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy
import scipy.signal
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Custom,
    Delay,
    Platform,
    Pulse,
    PulseSequence,
)
from scipy.optimize import least_squares
from scipy.signal import lfilter, lfilter_zi

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.protocols.ramsey.utils import fitting
from qibocal.protocols.utils import table_dict, table_html

FULL_WAVEFORM = np.concatenate([np.zeros(10), np.ones(90)])
"""Full waveform to be played."""
START = 10
"""Flux pulse start"""
SAMPLING_RATE = 1
"""Instrument sampling rate in GHz"""  # CONTROLLARE UNITA' DI MISURA


@dataclass
class CryoscopeParameters(Parameters):
    """Cryoscope runcard inputs."""

    duration_min: int
    """Minimum flux pulse duration."""
    duration_max: int
    """Maximum flux duration start."""
    duration_step: int
    """Flux pulse duration step."""
    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
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
    fir: dict[QubitId, list[float]] = field(default_factory=dict)
    """feedforward taps"""
    iir: dict[QubitId, list[float]] = field(default_factory=dict)
    """feedback taps"""

    # TODO: to be fixed
    def __contains__(self, key):
        return True


CryoscopeType = np.dtype([("duration", int), ("prob_1", np.float64)])
"""Custom dtype for Cryoscope."""


def generate_sequences(
    platform: Platform,
    qubit: QubitId,
    duration: int,
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
        envelope=Custom(i_=FULL_WAVEFORM[:duration], q_=np.zeros(duration)),
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
    flux_coefficients: dict[QubitId, list[float]] = field(default_factory=dict)
    """Flux - amplitude relation coefficients obtained from flux_amplitude_frequency routine"""
    data: dict[tuple[QubitId, str], npt.NDArray[CryoscopeType]] = field(
        default_factory=dict
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
    The delay between the two pi/2 pulses is fixed at t_max (maximum duration of flux pulse)
    + 100 ns (following the paper).
    """
    data = CryoscopeData(
        flux_pulse_amplitude=params.flux_pulse_amplitude,
    )

    for qubit in targets:
        assert (
            platform.calibration.single_qubits[qubit].qubit.flux_coefficients
            is not None
        ), "Cannot run cryoscope without flux coefficients, run cryoscope amplitude before the cryoscope"

        data.flux_coefficients[qubit] = platform.calibration.single_qubits[
            qubit
        ].qubit.flux_coefficients

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


def residuals(params, step_response, t):
    g, tau, exp_amplitude = params
    expmodel = step_response / (g * (1 + exp_amplitude * np.exp(-(t - START) / tau)))
    return expmodel - FULL_WAVEFORM[: len(t)]


def exponential_params(step_response, acquisition_time):
    init_guess = [1, 1, 1]
    t = np.arange(0, acquisition_time, 1)
    result = least_squares(residuals, init_guess, args=(step_response, t))
    return result.x


def filter_calc(params):
    g, tau, exp_amplitude = params
    alpha = 1 - np.exp(-1 / (SAMPLING_RATE * tau * (1 + exp_amplitude)))
    k = (
        exp_amplitude / ((1 + exp_amplitude) * (1 - alpha))
        if exp_amplitude < 0
        else exp_amplitude / (1 + exp_amplitude - alpha)
    )
    b0 = (1 - k + k * alpha) * g
    b1 = -(1 - k) * (1 - alpha) * g
    a0 = 1 * g
    a1 = -(1 - alpha) * g

    a = [a0, a1]  # feedback
    b = np.array([b0, b1])  # feedforward

    if np.max(b) >= 2:
        b = 2 * b / max(b)

    return a, b.tolist()


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
    time_decay = {}
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
        X_exp = 1 - 2 * data[(qubit, "MX")].prob_1
        Y_exp = 1 - 2 * data[(qubit, "MY")].prob_1

        norm_data = X_exp + 1j * Y_exp

        # demodulation frequency found by fitting sinusoidal
        demod_freq = -fitted_parameters[qubit, "MY"][2] / 2 / np.pi * sampling_rate

        # to be used in savgol_filter
        derivative_window_length = 7 / sampling_rate
        derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
        derivative_window_size += (derivative_window_size + 1) % 2

        # find demodulatation frequency
        demod_data = np.exp(2 * np.pi * 1j * x * demod_freq) * (norm_data)

        # compute phase
        phase = np.unwrap(np.angle(demod_data))

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

        # real detuning (reintroducing demod_freq)
        detuning[qubit] = (
            raw_detuning - demod_freq + sampling_rate * nyquist_order
        ).tolist()

        # params from flux_amplitude_frequency_protocol
        params = data.flux_coefficients[qubit]
        # params = [1.9412681243469971, -0.012534948170662627, 0.0005454772278201887]
        # params = [  # D2
        #     2.0578,
        #     -0.065,
        #     0.00147,
        # ]

        # invert frequency amplitude formula
        p = np.poly1d(params)
        amplitude[qubit] = [max((p - freq).roots).real for freq in detuning[qubit]]

        # compute step response
        step_response[qubit] = (
            np.array(amplitude[qubit]) / data.flux_pulse_amplitude
        ).tolist()

        # Derive IIR and FIR corrections
        acquisition_time = len(x)
        exp_params = exponential_params(step_response[qubit], acquisition_time)
        feedback_taps[qubit], feedforward_taps[qubit] = filter_calc(exp_params)
        _, time_decay[qubit], alpha[qubit] = exp_params

    return CryoscopeResults(
        amplitude=amplitude,
        detuning=detuning,
        step_response=step_response,
        fitted_parameters=fitted_parameters,
        exp_amplitude=alpha,
        tau=time_decay,
        fir=feedforward_taps,
        iir=feedback_taps,
    )


def _plot(data: CryoscopeData, fit: CryoscopeResults, target: QubitId):
    """Cryoscope plots."""

    fig = go.Figure()
    duration = data[(target, "MX")].duration

    fig.add_trace(
        go.Scatter(
            x=duration,
            y=FULL_WAVEFORM,
            name="exepcted waveform",
        ),
    )

    fitting_report = None

    if fit is not None:

        zi = (
            lfilter_zi(fit.fir[target], fit.iir[target]) * fit.step_response[target][0]
        )  # to review
        signal, _ = lfilter(
            fit.fir[target], fit.iir[target], fit.step_response[target], zi=zi
        )

        fig.add_trace(
            go.Scatter(
                x=duration,
                y=fit.step_response[target],
                name="derived waveform",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=duration,
                y=signal,
                name="filtered waveform",
            ),
        )

        exp_amplitude = fit.exp_amplitude[target]
        tau = fit.tau[target]
        fir = np.array(fit.fir[target])
        iir = np.array(fit.iir[target])

        fitting_report = table_html(
            table_dict(
                target,
                ["A", "tau", "FIR", "IIR"],
                [
                    (exp_amplitude,),
                    (tau,),
                    (fir,),
                    (iir,),
                ],
            )
        )

    return [fig], fitting_report


def _update(results: CryoscopeResults, platform: Platform, target: QubitId):
    update.feedback(results.fir[target], platform, target)
    update.feedforward(results.iir[target], platform, target)


cryoscope = Routine(_acquisition, _fit, _plot, _update)
