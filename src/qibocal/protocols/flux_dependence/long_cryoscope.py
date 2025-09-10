from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude

from ..utils import (
    HZ_TO_GHZ,
    extract_feature,
    readout_frequency,
)

__all__ = ["long_cryoscope"]


@dataclass
class LongCryoscopeParameters(Parameters):
    """LongCryoscope runcard inputs."""

    duration_min: float
    duration_max: float
    duration_step: float
    flux_pulse_amplitude: float
    flux_pulse_duration: float
    freq_width: float
    freq_step: float

    @property
    def frequency_range(self) -> np.ndarray:
        return np.arange(-self.freq_width / 2, self.freq_width / 2, self.freq_step)

    @property
    def duration_range(self) -> np.ndarray:
        return np.arange(self.duration_min, self.duration_max, self.duration_step)


@dataclass
class LongCryoscopeResults(Results):
    """LongCryoscope outputs."""

    fitting_parameters: dict[QubitId, list]


@dataclass
class LongCryoscopeData(Data):
    """LongCryoscope acquisition outputs."""

    frequency_swept: dict[QubitId, list]
    duration_swept: list
    data: dict[QubitId, np.ndarray] = field(default_factory=dict)

    def grid(self, qubit: QubitId) -> tuple[np.ndarray]:
        x, y = np.meshgrid(self.frequency_swept[qubit], self.duration_swept)
        return x.ravel(), y.ravel(), magnitude(self.data[qubit]).ravel()


def _acquisition(
    params: LongCryoscopeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> LongCryoscopeData:
    """Data acquisition for LongCryoscope Experiment."""

    sequence = PulseSequence()
    freq_sweepers = []
    delays = [Delay(duration=0) for _ in range(2 * len(targets))]
    for i, q in enumerate(targets):
        natives = platform.natives.single_qubit[q]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]
        flux_channel = platform.qubits[q].flux
        flux_pulse = Pulse(
            duration=params.flux_pulse_duration,
            amplitude=params.flux_pulse_amplitude,
            envelope=Rectangular(),
        )

        sequence.append((flux_channel, flux_pulse))
        sequence.align([flux_channel, qd_channel, ro_channel])
        sequence.append((qd_channel, delays[2 * i]))
        sequence.append((qd_channel, qd_pulse))
        sequence.append((ro_channel, delays[2 * i + 1]))
        sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((ro_channel, ro_pulse))

        freq_sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                # we should correct this including the expected shift due to the flux pulse
                values=platform.config(qd_channel).frequency + params.frequency_range,
                channels=[qd_channel],
            )
        )

    duration_sweeper = Sweeper(
        parameter=Parameter.duration,
        values=params.duration_range,
        pulses=delays,
    )

    data = LongCryoscopeData(
        frequency_swept={
            qubit: freq_sweepers[i].values.tolist() for i, qubit in enumerate(targets)
        },
        duration_swept=duration_sweeper.values.tolist(),
    )
    results = platform.execute(
        [sequence],
        [[duration_sweeper], freq_sweepers],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in targets
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for qubit in targets:
        acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[-1].id
        data.data[qubit] = results[acq_handle]
    return data


def _fit(data: LongCryoscopeData) -> LongCryoscopeResults:
    fitting_parameters = {}

    return LongCryoscopeResults(fitting_parameters=fitting_parameters)


def _plot(data: LongCryoscopeData, fit: LongCryoscopeResults, target: QubitId):
    """Plotting function for LongCryoscope Experiment."""
    fig = go.Figure()
    freq, delay = extract_feature(*data.grid(target), find_min=False)

    fig.add_trace(
        go.Heatmap(
            x=data.duration_swept,
            y=np.array(data.frequency_swept[target]) * HZ_TO_GHZ,
            z=magnitude(data.data[target]).T,
            colorbar=dict(title="Signal [a.u.]"),
            colorscale="Viridis",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=delay,
            y=freq * HZ_TO_GHZ,
            mode="markers",
            showlegend=True,
            name="Extract feature",
            marker=dict(color="rgb(248, 248, 248)"),
        )
    )

    fig.update_layout(
        xaxis_title="Delay [ns]",
        yaxis_title="Frequency [GHz]",
        showlegend=True,
        legend=dict(orientation="h"),
    )
    return [fig], ""


def _update(
    results: LongCryoscopeResults, platform: CalibrationPlatform, qubit: QubitId
):
    pass


long_cryoscope = Routine(_acquisition, _fit, _plot, _update)
"""LongCryoscope Routine object."""
