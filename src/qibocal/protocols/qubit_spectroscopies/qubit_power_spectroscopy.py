from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform

from ...update import replace
from ..resonator_spectroscopies.resonator_punchout import ResonatorPunchoutData
from ..utils import HZ_TO_GHZ, Range, readout_frequency, to_range
from .qubit_spectroscopy import QubitSpectroscopyResults

__all__ = ["qubit_power_spectroscopy"]


@dataclass
class QubitPowerSpectroscopyParameters(Parameters):
    """QubitPowerSpectroscopy runcard inputs."""

    frequency: Range | None = None
    """Frequencies for the sweep [Hz]."""
    freq_width: int | None = None
    """Width for frequency sweep relative  to the drive frequency [Hz]."""
    freq_step: int | None = None
    """Frequency step for sweep [Hz]."""
    amplitude: Range | None = None
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
        try:
            qd_channel = platform.qubits[q].drive
            center = platform.config(qd_channel).frequency
        except KeyError:
            center = 0.0
        assert isinstance(center, float)
        return (
            to_range(self.frequency, center=center)
            if self.frequency is not None
            else (
                center - self.freq_width / 2,
                center + self.freq_width / 2,
                self.freq_step,
            )
        )

    def amplitude_range(self) -> Range:
        return (
            to_range(self.amplitude, center=0.0)
            if self.amplitude is not None
            else (self.min_amp, self.max_amp, self.step_amp)
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
        # average signal, phase, i and q over the number of shots defined in the runcard
        data.data[qubit] = results[ro_pulse.id]

    return data


def _fit(data: QubitPowerSpectroscopyData) -> Results:
    """Do not perform any fitting procedure."""
    return Results()


def _plot(
    data: ResonatorPunchoutData,
    target: QubitId,
    fit: QubitSpectroscopyResults | None = None,
):
    """Plot QubitPunchout."""
    figures = []
    fitting_report = ""
    fig = go.Figure()
    x, y, _ = data.grid(target)
    fig.add_trace(
        go.Heatmap(
            x=x * HZ_TO_GHZ,
            y=y,
            z=data.normalized_signal(target).ravel(),
            colorbar=dict(title="Normalized signal"),
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(title_text="Drive frequency [GHz]")
    fig.update_yaxes(title_text="Drive amplitude [a.u.]")

    figures.append(fig)

    return figures, fitting_report


qubit_power_spectroscopy = Protocol(_acquisition, _fit, _plot)
"""QubitPowerSpectroscopy Protocol object."""
