from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.fitting.classifier.qubit_fit import QubitFit

from ..classification import ClassificationType


@dataclass
class QubitReadoutFrequencyParameters(Parameters):
    """ResonatorAmplitude runcard inputs."""

    resonator_freq_width: float
    """Amplituude step to be probed."""
    resonator_freq_step: float
    """Amplitude start."""
    flux_pulse_amplitude_min: float
    """Amplitude stop value"""
    flux_pulse_amplitude_max: float
    """Probability error threshold to stop the best amplitude search"""
    flux_pulse_amplitude_step: float


@dataclass
class QubitReadoutFrequencyData(Data):
    """Data class for `resoantor_amplitude` protocol."""

    data: dict[tuple, npt.NDArray[ClassificationType]] = field(default_factory=dict)
    amplitude: dict[QubitId, list] = field(default_factory=dict)
    frequency: dict[QubitId, list] = field(default_factory=dict)


@dataclass
class QubitReadoutFrequencyResults(Results):
    """Result class for `resonator_amplitude` protocol."""

    assignment_fidelity: dict[QubitId, list] = field(default_factory=dict)


def _acquisition(
    params: QubitReadoutFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitReadoutFrequencyData:
    r"""
    Data acquisition for resoantor amplitude optmization.
    This protocol sweeps the readout amplitude performing a classification routine
    and evaluating the error probability at each step. The sweep will be interrupted
    if the probability error is less than the `error_threshold`.

    Args:
        params (:class:`ResonatorAmplitudeParameters`): input parameters
        platform (:class:`CalibrationPlatform`): Qibolab's platform
        targets (list): list of QubitIds to be characterized

    Returns:
        data (:class:`ResonatorAmplitudeData`)
    """

    native = platform.natives.single_qubit

    delta_frequency_range = np.arange(
        -params.resonator_freq_width / 2,
        params.resonator_freq_width / 2,
        params.resonator_freq_step,
    )
    freq_sweeepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=platform.config(platform.qubits[q].probe).frequency
            + delta_frequency_range,
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]

    data = QubitReadoutFrequencyData()
    for qubit in targets:
        data.amplitude[qubit] = np.arange(
            params.flux_pulse_amplitude_min,
            params.flux_pulse_amplitude_max,
            params.flux_pulse_amplitude_step,
        ).tolist()
        data.frequency[qubit] = (
            platform.config(platform.qubits[qubit].probe).frequency
            + delta_frequency_range
        ).tolist()
    for state in [0, 1]:
        ro_pulses = {}
        flux_pulses = []
        sequence = PulseSequence()
        for q in targets:
            ro_sequence = native[q].MZ()
            ro_pulses[q] = ro_sequence[0][1].id
            flux_channel = platform.qubits[q].flux
            flux_pulse = Pulse(
                duration=ro_sequence.duration,
                amplitude=params.flux_pulse_amplitude_max / 2,
                envelope=Rectangular(),
            )
            flux_pulses.append(flux_pulse)
            ro_sequence.append((flux_channel, flux_pulse))
            sequence += ro_sequence

        if state == 1:
            rx_sequence = PulseSequence()
            for q in targets:
                rx_sequence += native[q].RX()
            sequence = rx_sequence | sequence

        amp_sweepers = Sweeper(
            parameter=Parameter.amplitude,
            range=(
                params.flux_pulse_amplitude_min,
                params.flux_pulse_amplitude_max,
                params.flux_pulse_amplitude_step,
            ),
            pulses=flux_pulses,
        )

        results = platform.execute(
            [sequence],
            [[amp_sweepers], freq_sweeepers],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        )
        for qubit in targets:
            serial = ro_pulses[qubit]
            result = results[serial]
            data.register_qubit(
                ClassificationType,
                (qubit, state),
                dict(
                    i=result[..., 0],
                    q=result[..., 1],
                ),
            )
            list(data.data)

    return data


def _fit(data: QubitReadoutFrequencyData) -> QubitReadoutFrequencyResults:

    assignment_fidelity = {}
    for qubit in data.qubits:
        assignment_fidelity[qubit] = []
        state0_data = data.data[qubit, 0]
        state1_data = data.data[qubit, 1]
        for i in range(len(data.amplitude[qubit])):
            for j in range(len(data.frequency[qubit])):
                state0 = np.column_stack(
                    (state0_data[:, i, j].i, state0_data[:, i, j].q)
                )
                state1 = np.column_stack(
                    (state1_data[:, i, j].i, state1_data[:, i, j].q)
                )
                nshots = len(state0)
                iq_values = np.concatenate((state0, state1))
                states = [0] * nshots + [1] * nshots
                model = QubitFit()
                model.fit(iq_values, np.array(states))
                assignment_fidelity[qubit].append(model.assignment_fidelity)

    # TODO: select best point
    return QubitReadoutFrequencyResults(assignment_fidelity=assignment_fidelity)


def _plot(
    data: QubitReadoutFrequencyData, fit: QubitReadoutFrequencyResults, target: QubitId
):
    """Plotting function for Optimization RO amplitude."""
    # print(data.data["B4"].i[:0].shape)
    figures = []
    # opacity = 1
    fitting_report = None
    fig = make_subplots(
        rows=1,
        cols=1,
    )

    frequency, amplitude = np.meshgrid(data.frequency[target], data.amplitude[target])

    if fit is not None:
        fig.add_trace(
            go.Heatmap(
                x=amplitude.ravel(),
                y=frequency.ravel() / 1e9,
                z=fit.assignment_fidelity[target],
            ),
            row=1,
            col=1,
        )

    figures.append(fig)
    fitting_report = ""
    fig.update_layout(
        xaxis_title="Flux pulse amplitude [a.u.]",
        yaxis_title="Readout frequency [GHz]",
    )
    return figures, fitting_report


def _update(
    results: QubitReadoutFrequencyResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    """Update function for qubit readout frequency protocol."""
    # TODO: Add flux pulse to readout sequence
    # TODO: Add attribute qubit readout frequency in calibration somewhere


qubit_readout_frequency = Routine(_acquisition, _fit, _plot, _update)
"""Resonator Amplitude Routine  object."""
