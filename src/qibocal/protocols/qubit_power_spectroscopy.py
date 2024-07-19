from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Parameters, Results, Routine

from .qubit_spectroscopy import QubitSpectroscopyResults
from .resonator_punchout import ResonatorPunchoutData
from .utils import HZ_TO_GHZ


@dataclass
class QubitPowerSpectroscopyParameters(Parameters):
    """QubitPowerSpectroscopy runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the drive frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    duration: int
    """Drive duration."""
    amplitude: Optional[float] = None
    """Initial drive amplitude."""


@dataclass
class QubitPowerSpectroscopyData(ResonatorPunchoutData):
    """QubitPowerSpectroscopy data acquisition."""


def _acquisition(
    params: QubitPowerSpectroscopyParameters,
    platform: Platform,
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
    amplitudes = {}
    for qubit in targets:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.duration
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        if params.amplitude is not None:
            qd_pulses[qubit].amplitude = params.amplitude
        amplitudes[qubit] = qd_pulses[qubit].amplitude

        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    # drive frequency
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [qd_pulses[qubit] for qubit in targets],
        type=SweeperType.OFFSET,
    )
    # drive amplitude
    amplitude_range = np.arange(
        params.min_amp_factor, params.max_amp_factor, params.step_amp_factor
    )
    amp_sweeper = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        [qd_pulses[qubit] for qubit in targets],
        type=SweeperType.FACTOR,
    )

    # data
    data = QubitPowerSpectroscopyData(
        amplitudes=amplitudes,
        resonator_type=platform.resonator_type,
    )

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        amp_sweeper,
        freq_sweeper,
    )

    # retrieve the results for every qubit
    for qubit, ro_pulse in ro_pulses.items():
        # average signal, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulse.serial]
        data.register_qubit(
            qubit,
            signal=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + qd_pulses[qubit].frequency,
            amp=amplitude_range * amplitudes[qubit],
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
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Signal [a.u.]",
            "phase [rad]",
        ),
    )
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    amplitudes = qubit_data.amp

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=amplitudes,
            z=qubit_data.signal,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=amplitudes,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(title_text="Drive frequency [GHz]", row=1, col=1)
    fig.update_xaxes(title_text="Drive frequency [GHz]", row=1, col=2)
    fig.update_yaxes(title_text="Drive amplitude [a.u.]", row=1, col=1)

    figures.append(fig)

    return figures, fitting_report


qubit_power_spectroscopy = Routine(_acquisition, _fit, _plot)
"""QubitPowerSpectroscopy Routine object."""
