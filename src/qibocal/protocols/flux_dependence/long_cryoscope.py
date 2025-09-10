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
    GHZ_TO_HZ,
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


@dataclass
class LongCryoscopeData(Data):
    """LongCryoscope acquisition outputs."""

    frequency_swept: dict[QubitId, list]
    duration_swept: list
    data: dict[QubitId, np.ndarray] = field(default_factory=dict)

    def grid(self, qubit: QubitId) -> tuple[np.ndarray]:
        x, y = np.meshgrid(self.frequency_swept[qubit], self.duration_swept)
        return x.ravel(), y.ravel(), magnitude(self.data[qubit]).ravel()


def sequence(platform, target, flux_pulse_amplitude, delay):
    seq = PulseSequence()
    natives = platform.natives.single_qubit[target]
    qd_channel, qd_pulse = natives.RX()[0]
    ro_channel, ro_pulse = natives.MZ()[0]
    flux_channel = platform.qubits[target].flux
    flux_pulse = Pulse(
        duration=2 * delay + qd_pulse.duration,
        amplitude=flux_pulse_amplitude,
        envelope=Rectangular(),
    )

    seq.append((flux_channel, flux_pulse))
    seq.append((qd_channel, Delay(duration=delay)))
    seq.append((qd_channel, qd_pulse))
    seq.append((ro_channel, Delay(duration=flux_pulse.duration)))
    seq.append((ro_channel, ro_pulse))
    return seq


def _acquisition(
    params: LongCryoscopeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> LongCryoscopeData:
    """Data acquisition for LongCryoscope Experiment."""

    freq_sweepers = []
    data_ = {q: [] for q in targets}
    for delay in params.duration_range:
        seq = PulseSequence()
        for i, q in enumerate(targets):
            seq += sequence(platform, q, params.flux_pulse_amplitude, delay)
            qd_channel = platform.qubits[q].drive
            freq_sweepers.append(
                Sweeper(
                    parameter=Parameter.frequency,
                    values=platform.config(qd_channel).frequency
                    + platform.calibration.single_qubits[q].qubit.detuning(
                        params.flux_pulse_amplitude
                    )
                    * GHZ_TO_HZ
                    + params.frequency_range,
                    channels=[qd_channel],
                )
            )

        results = platform.execute(
            [seq],
            [freq_sweepers],
            updates=[
                {
                    platform.qubits[q].probe: {
                        "frequency": readout_frequency(q, platform)
                    }
                }
                for q in targets
            ],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        for qubit in targets:
            acq_handle = list(seq.channel(platform.qubits[qubit].acquisition))[-1].id
            data_[qubit].append(results[acq_handle])
    data = LongCryoscopeData(
        frequency_swept={
            qubit: freq_sweepers[i].values.tolist() for i, qubit in enumerate(targets)
        },
        duration_swept=params.duration_range.tolist(),
        data={qubit: np.stack(result) for qubit, result in data_.items()},
    )
    return data


def _fit(data: LongCryoscopeData) -> LongCryoscopeResults:
    return LongCryoscopeResults()


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
