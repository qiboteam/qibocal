"""Flux pulse reconstruction experiment"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy
import scipy.signal
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Platform,
    Pulse,
    PulseSequence,
    Rectangular,
)

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.protocols.ramsey.utils import fitting

# TODO: remove hard-coded QM parameters
FEEDFORWARD_MAX = 2 - 2**-16
"""Maximum feedforward tap value"""
FEEDBACK_MAX = 1 - 2**-20
"""Maximum feedback tap value"""
SAMPLING_RATE = 1
"""Instrument sampling rate in GSamples"""


@dataclass
class FluxReconstructionParameters(Parameters):
    """Flux reconstruction outputs."""

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
class FluxReconstructionResults(Results):
    """Flux reconstruction outputs."""

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


FluxReconstructionType = np.dtype([("duration", int), ("prob_1", np.float64)])
"""Custom dtype for Flux Reconstruction."""


def generate_sequences(
    platform: Platform,
    qubit: QubitId,
    duration: int,
    params: FluxReconstructionParameters,
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
class FluxReconstructionData(Data):
    """Flux reconstruction acquisition outputs."""

    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    flux_coefficients: dict[QubitId, list[float]] = field(default_factory=dict)
    """Flux - amplitude relation coefficients obtained from flux_amplitude_frequency routine"""
    data: dict[tuple[QubitId, str], npt.NDArray[FluxReconstructionType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: FluxReconstructionParameters,
    platform: Platform,
    targets: list[QubitId],
) -> FluxReconstructionData:
    """Acquisition for flux reconstruction experiment, is the same of cryoscope experiment.

    The following sequence is played for each qubit.

    drive    --- RY90 ------------------- RY90 -------
    flux     --------- FluxPulse(t) ------------------
    readout  ----------------------------------- MZ --

    The previous sequence measures <X>, to measure <Y> the second drive pulse
    is replaced with RX90.
    The delay between the two pi/2 pulses is fixed at t_max (maximum duration of flux pulse)
    + 100 ns (following the paper).
    """
    data = FluxReconstructionType(
        flux_pulse_amplitude=params.flux_pulse_amplitude,
    )

    for qubit in targets:
        assert (
            platform.calibration.single_qubits[qubit].qubit.flux_coefficients
            is not None
        ), f"Cannot run cryoscope without flux coefficients, run cryoscope amplitude on qubit {qubit} before the flux reconstruction"

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
                    FluxReconstructionType,
                    (qubit, measure),
                    dict(
                        duration=np.array([duration]),
                        prob_1=result,
                    ),
                )

    return data


def _fit(data: FluxReconstructionData) -> FluxReconstructionResults:
    """Postprocessing for flux reconstruction experiment.

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

        # invert frequency amplitude formula
        p = np.poly1d(params)
        amplitude[qubit] = [max((p - freq).roots).real for freq in detuning[qubit]]

        # compute step response
        step_response[qubit] = (
            np.array(amplitude[qubit]) / data.flux_pulse_amplitude
        ).tolist()

    return FluxReconstructionResults(
        amplitude=amplitude,
        detuning=detuning,
        step_response=step_response,
    )


def _plot(
    data: FluxReconstructionData, fit: FluxReconstructionResults, target: QubitId
):
    """Flux reconstruction plots."""

    fig = go.Figure()
    duration = data[(target, "MX")].duration

    fitting_report = None
    if fit is not None:

        fig.add_trace(
            go.Scatter(
                x=duration,
                y=fit.step_response[target],
                name="Reconstructed flux waveform",
            ),
        )

        fitting_report = None

    return [fig], fitting_report


flux_reconstruction = Routine(_acquisition, _fit, _plot)
