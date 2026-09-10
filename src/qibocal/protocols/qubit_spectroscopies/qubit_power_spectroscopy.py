from dataclasses import dataclass
from typing import cast

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    IqConfig,
    Parameter,
    PulseSequence,
    Sweeper,
)
from sklearn.decomposition import PCA

from qibocal.auto.operation import Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform

from ...result import magnitude, phase
from ...update import replace
from ..resonator_spectroscopies.resonator_punchout import ResonatorPunchoutData
from ..utils import HZ_TO_GHZ, Range, RangeLike, readout_frequency, to_range
from .qubit_spectroscopy import QubitSpectroscopyResults

__all__ = ["qubit_power_spectroscopy"]

PCA_VARIANCE_THRESHOLD = 0.85
"""Minimum explained variance of the first PCA component to show the PCA heatmap.

If the first component explains less than this fraction of the total variance,
the signal magnitude and phase are shown in 2D subplots instead.
"""


@dataclass
class QubitPowerSpectroscopyParameters(Parameters):
    """QubitPowerSpectroscopy runcard inputs."""

    frequency: RangeLike | None = None
    """Frequencies for the sweep [Hz]."""
    freq_width: int | None = None
    """Width for frequency sweep relative  to the drive frequency [Hz]."""
    freq_step: int | None = None
    """Frequency step for sweep [Hz]."""
    amplitude: RangeLike | None = None
    """Amplitudes for the sweep [a.u.]."""
    min_amp: float | None = None
    """Minimum amplitude."""
    max_amp: float | None = None
    """Maximum amplitude."""
    step_amp: float | None = None
    """Step amplitude."""
    duration: int = 4000
    """Drive duration."""

    def frequency_range(self, q: QubitId, platform: CalibrationPlatform) -> Range:
        qd_channel = platform.qubits[q].drive
        assert qd_channel is not None
        center = cast(IqConfig, platform.config(qd_channel)).frequency

        def legacy_range() -> Range:
            assert self.freq_width is not None
            assert self.freq_step is not None
            return (
                center - self.freq_width / 2,
                center + self.freq_width / 2,
                self.freq_step,
            )

        assert isinstance(center, float)
        return (
            to_range(self.frequency, center=center)
            if self.frequency is not None
            else legacy_range()
        )

    def amplitude_range(self) -> Range:
        def legacy_range() -> Range:
            assert self.min_amp is not None
            assert self.max_amp is not None
            assert self.step_amp is not None
            return (self.min_amp, self.max_amp, self.step_amp)

        return (
            to_range(self.amplitude) if self.amplitude is not None else legacy_range()
        )


@dataclass
class QubitPowerSpectroscopyData(ResonatorPunchoutData):
    """QubitPowerSpectroscopy data acquisition."""


def _acquisition(
    params: QubitPowerSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitPowerSpectroscopyData:
    """Perform a qubit spectroscopy experiment with different amplitudes.

    For high amplitude it should be possible to see more peaks: corresponding to
    the (0-2)/2 frequency and the 1-2.
    This experiment can be used also to test if a peak is a qubit: if it is, the
    peak will get larger while increasing the power of the drive.
    """
    # define the sequence: RX - MZ
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    freq_sweepers = {}
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        qd_pulse = replace(qd_pulse, duration=params.duration)

        qd_pulses[qubit] = qd_pulse
        ro_pulses[qubit] = ro_pulse

        sequence.append((qd_channel, qd_pulse))
        sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((ro_channel, ro_pulse))

        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            range=params.frequency_range(qubit, platform),
            channels=[qd_channel],
        )

    amp_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=params.amplitude_range(),
        pulses=[qd_pulses[qubit] for qubit in targets],
    )

    # data
    data = QubitPowerSpectroscopyData(
        resonator_type=platform.resonator_type,
        amplitudes=amp_sweeper.values.tolist(),
        frequencies={qubit: freq_sweepers[qubit].values.tolist() for qubit in targets},
    )

    results = platform.execute(
        [sequence],
        [[amp_sweeper], [freq_sweepers[q] for q in targets]],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in targets
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # retrieve the results for every qubit
    for qubit, ro_pulse in ro_pulses.items():
        # average i and q
        data.data[qubit] = results[ro_pulse.id]

    return data


def _fit(data: QubitPowerSpectroscopyData) -> Results:
    """Do not perform any fitting procedure."""
    return Results()


def _heatmap_figure(
    frequencies: np.ndarray,
    amplitudes: list,
    matrix: np.ndarray,
    colorbar_title: str,
) -> go.Figure:
    """Build a 2D heatmap of ``matrix`` (shape ``(n_amplitudes, n_frequencies)``)."""
    fig = go.Figure(
        go.Heatmap(
            x=frequencies,
            y=amplitudes,
            z=matrix,
            colorbar={"title": colorbar_title},
            colorscale="Viridis",
        )
    )
    fig.update_xaxes(title_text="Drive frequency [GHz]")
    fig.update_yaxes(title_text="Drive amplitude [a.u.]")

    return fig


def _signal_phase_figure(
    frequencies: np.ndarray,
    amplitudes: list,
    raw: np.ndarray,
) -> go.Figure:
    """Build a figure with signal magnitude and phase in 2 stacked subplots."""
    shape = (len(amplitudes), len(frequencies))
    signal_matrix = magnitude(raw).reshape(shape)
    phase_matrix = phase(raw).reshape(shape)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=amplitudes,
            z=signal_matrix,
            colorbar={"title": "Signal magnitude"},
            colorscale="Viridis",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=amplitudes,
            z=phase_matrix,
            colorbar={"title": "Signal phase [rad]"},
            colorscale="Viridis",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Drive frequency [GHz]", row=2, col=1)
    fig.update_yaxes(title_text="Drive amplitude [a.u.]", row=1, col=1)
    return fig


def _plot(
    data: ResonatorPunchoutData,
    target: QubitId,
    fit: QubitSpectroscopyResults | None = None,
):
    """Plot QubitPowerSpectroscopy.

    A single 2D figure is shown: the PCA-transformed signal if the first
    principal component explains most of the variance, otherwise the signal
    magnitude and phase in two subplots.
    """
    frequencies = np.asarray(data.frequencies[target]) * HZ_TO_GHZ
    amplitudes = data.amplitudes
    raw = data.data[target]

    # first principal component of the IQ signal at each frequency
    pc_matrix = np.asarray([PCA().fit_transform(x)[:, 0] for x in raw])

    # the first component explains most of the variance -> a single 1D
    # projection is representative, so show the PCA heatmap
    first_component_variance = float(
        PCA().fit(raw.reshape(-1, raw.shape[-1])).explained_variance_ratio_[0]
    )
    if first_component_variance > PCA_VARIANCE_THRESHOLD:
        figure = _heatmap_figure(
            frequencies, amplitudes, pc_matrix, "Normalized signal"
        )
    else:
        figure = _signal_phase_figure(frequencies, amplitudes, raw)

    return [figure], ""


qubit_power_spectroscopy = Protocol(_acquisition, _fit, _plot)
"""QubitPowerSpectroscopy Protocol object."""
