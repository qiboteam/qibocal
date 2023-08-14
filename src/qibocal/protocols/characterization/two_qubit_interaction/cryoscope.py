"""Cryoscope experiment, corrects distortions."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.platform import Platform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Qubits, Results


@dataclass
class CryoscopeParameters(Parameters):
    """Cryoscope runcard inputs."""

    amplitude_min: float
    """Minimum flux pulse amplitude."""
    amplitude_max: float
    """Maximum flux pulse amplitude."""
    amplitude_step: float
    """Flux pulse amplitude step."""
    duration_min: int
    """Minimum flux pulse duration."""
    duration_max: int
    """Maximum flux duration start."""
    duration_step: int
    """Flux pulse duration step."""

    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    dt: Optional[int] = 0
    """Time delay between flux pulses and readout."""
    nshots: Optional[int] = None
    """Number of shots per point."""

    # flux_pulse_shapes
    # TODO support different shapes, for now only rectangular


@dataclass
class CryoscopeResults(Results):
    """Cryoscope outputs."""

    distorted_pulses: dict[QubitId, npt.NDArray[np.float64]]
    """Reconstructed distorted pulses."""
    ideal_pulses: dict[QubitId, npt.NDArray[np.float64]]
    """Ideally sent pulses."""


# TODO
# not really a probability as for now
CryoscopeType = np.dtype(
    [("amp", np.float64), ("duration", np.float64), ("prob", np.float64)]
)
"""Custom dtype for Cryoscope."""


@dataclass
class CryoscopeData(Data):
    """Cryoscope acquisition outputs."""

    data: dict[QubitId, str, npt.NDArray[CryoscopeType]] = field(default_factory=dict)

    def register_qubit(
        self,
        qubit: QubitId,
        tag: str,
        amps: npt.NDArray[np.float64],
        durs: npt.NDArray[np.int32],
        probs: npt.NDArray[np.float64],
    ):
        """Store output for a single qubit."""
        size = len(amps) * len(durs)
        amplitudes, durations = np.meshgrid(amps, durs)
        ar = np.empty(size, dtype=CryoscopeType)
        ar["amp"] = amplitudes.ravel()
        ar["duration"] = durations.ravel()
        ar["prob"] = probs.ravel()

        self.data[qubit, tag] = np.rec.array(ar)

    def __getitem__(self, qubit):
        return {
            index: value
            for index, value in self.data.items()
            if set(qubit).issubset(index)
        }

    def get_signal(self, qubit, amp=None):
        mx_data = self.data[qubit, "MX"]
        my_data = self.data[qubit, "MY"]

        if amp is not None:
            index = mx_data.index(amp)
            return mx_data.prob[index] + 1j * my_data.prob[index]

        return mx_data + 1j * my_data


def _acquisition(
    params: CryoscopeParameters,
    platform: Platform,
    qubits: Qubits,
) -> CryoscopeData:
    # define sequences of pulses to be executed
    MX_seq = PulseSequence()
    MY_seq = PulseSequence()

    initial_RY90_pulses = {}
    flux_pulses = {}
    RX90_pulses = {}
    RY90_pulses = {}
    MZ_ro_pulses = {}

    for qubit in qubits:
        # start at |+> by applying a Ry(90)
        initial_RY90_pulses[qubit] = platform.create_RX90_pulse(
            qubit, start=0, relative_phase=np.pi / 2
        )

        # TODO add support for flux pulse shapes
        # if params.flux_pulse_shapes and len(params.flux_pulse_shapes) == len(qubits):
        #     flux_pulse_shape = eval(params.flux_pulse_shapes[qubit])
        # else:
        #     flux_pulse_shape = Rectangular()
        flux_pulse_shape = Rectangular()

        # apply a detuning flux pulse
        flux_pulses[qubit] = FluxPulse(
            start=initial_RY90_pulses[qubit].finish,
            duration=params.duration_min,
            amplitude=params.flux_pulse_amplitude,
            shape=flux_pulse_shape,
            channel=qubits[qubit].flux,
            qubit=qubit,
        )

        # wait delay_before_readout
        # rotate around the X axis RX(-pi/2) to measure Y component
        RX90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=flux_pulses[qubit].finish + params.dt,
            relative_phase=np.pi,
        )

        # wait delay_before_readout
        # rotate around the Y axis RX(-pi/2) to measure X component
        RY90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=flux_pulses[qubit].finish + params.dt,
            relative_phase=np.pi / 2,
        )

        # add readout at the end of the sequences
        MZ_ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses[qubit].finish
        )

        # create the sequences
        MX_seq.add(
            initial_RY90_pulses[qubit],
            flux_pulses[qubit],
            RY90_pulses[qubit],  # rotate around Y to measure X
            MZ_ro_pulses[qubit],
        )
        MY_seq.add(
            initial_RY90_pulses[qubit],
            flux_pulses[qubit],
            RX90_pulses[qubit],  # rotate around X to measure Y
            MZ_ro_pulses[qubit],
        )

    amplitude_range = np.arange(
        params.amplitude_min, params.amplitude_max, params.amplitude_step
    )
    duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    amp_sweeper = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        pulses=list(flux_pulses.values()),
        type=SweeperType.ABSOLUTE,
    )

    dur_sweeper = Sweeper(
        Parameter.duration,
        duration_range,
        pulses=list(flux_pulses.values()),
        type=SweeperType.ABSOLUTE,
    )

    MX_tag = "MX"
    MY_tag = "MY"

    options = ExecutionParameters(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = CryoscopeData()
    for sequence, tag in [(MX_seq, MX_tag), (MY_seq, MX_tag)]:
        results = platform.sweep(sequence, options, amp_sweeper, dur_sweeper)

        for qubit in qubits:
            result = results[MZ_ro_pulses[qubit].serial]

            prob = result.magnitude

            data.register_qubit(qubit, tag, amplitude_range, duration_range, prob)


def _fit(data: CryoscopeData) -> CryoscopeResults:
    distorted_pulses = {}
    ideal_pulses = {}

    qubits = data.qubits

    for qubit in qubits:
        data_q = data[qubit]
        durations = data_q.duration
        amplitudes = data_q.amp

        num_points = durations.shape[0]
        interval = (durations[1] - durations[0]) * 1e-9  # TODO check why
        amp_freqs = np.fft.fftfreq(n=num_points, d=interval)
        mask = np.argsort(amp_freqs)
        amp_freqs = amp_freqs[mask]

        # TODO I am copying from old implementation, but here there are a lot of hardcoded stuff
        sampling_rate = 1 / (durations[1] - durations[0])
        derivative_window_length = 7 / sampling_rate
        derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
        derivative_window_size += (derivative_window_size + 1) % 2
        derivative_window_size = 15
        derivative_order = 3
        nyquist_order = 0

    return CryoscopeResults(distorted_pulses, ideal_pulses)


def _plot(data: CryoscopeData, fit: CryoscopeResults, qubit: QubitId):
    """Cryoscope plots."""

    result = data[qubit]

    figures = []

    # first it's a figure "comparison" of ideal pulse Vs distorted
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.duration, y=fit.distorted_pulses[qubit], name="Distorted pulse"
        ),
    )
    fig.add_trace(
        go.Scatter(x=result.duration, y=fit.ideal_pulses[qubit], name="Ideal pulse"),
    )
    figures.append(fig)

    fitting_report = f"Cryoscope of qubit {qubit}"

    return figures, fitting_report
