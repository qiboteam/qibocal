from dataclasses import dataclass
from typing import Optional

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

from qibocal.auto.operation import Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from ...result import magnitude
from ...update import replace
from ..resonator_spectroscopies.resonator_punchout import ResonatorPunchoutData
from ..utils import HZ_TO_GHZ, readout_frequency
from .qubit_spectroscopy import QubitSpectroscopyResults

__all__ = ["qubit_power_spectroscopy"]


@dataclass
class QubitPowerSpectroscopyParameters(Parameters):
    """QubitPowerSpectroscopy runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the drive frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    min_amp: float
    """Minimum amplitude."""
    max_amp: float
    """Maximum amplitude."""
    step_amp: float
    """Step amplitude."""
    duration: int
    """Drive duration."""


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
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )
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

        f0 = platform.config(qd_channel).frequency
        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            values=f0 + delta_frequency_range,
            channels=[qd_channel],
        )

    amp_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.min_amp, params.max_amp, params.step_amp),
        pulses=[qd_pulses[qubit] for qubit in targets],
    )

    # data
    data = QubitPowerSpectroscopyData(
        resonator_type=platform.resonator_type,
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
        result = results[ro_pulse.id]
        data.register_qubit(
            qubit,
            signal=magnitude(result),
            freq=freq_sweepers[qubit].values,
            amp=amp_sweeper.values,
        )

    return data


def _fit(data: QubitPowerSpectroscopyData) -> Results:
    """Do not perform any fitting procedure."""
    return Results()


def _plot(
    data: ResonatorPunchoutData,
    target: QubitId,
    fit: Optional[QubitSpectroscopyResults] = None,
):
    """Plot QubitPunchout."""
    figures = []
    fitting_report = ""
    fig = go.Figure()
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    amplitudes = qubit_data.amp

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=amplitudes,
            z=qubit_data.signal,
            colorbar_x=0.46,
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


qubit_power_spectroscopy = Routine(_acquisition, _fit, _plot)
"""QubitPowerSpectroscopy Routine object."""
