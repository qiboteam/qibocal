from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from .utils import HZ_TO_GHZ, V_TO_UV


@dataclass
class ResonatorQubitSpectroscopyParameters(Parameters):
    """ResonatorQubitSpectroscopy runcard inputs."""

    drive_freq_width: int
    """Width [Hz] for frequency sweep relative to the drive frequency."""
    drive_freq_step: int
    """Frequency [Hz] step for drive frequency sweep."""
    readout_freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency."""
    readout_freq_step: int
    """Frequency [Hz] step for readout frequency sweep."""
    drive_duration: int
    """Drive pulse duration [ns]. Same for all qubits."""
    drive_amplitude: Optional[float] = None
    """Drive pulse amplitude (optional). Same for all qubits."""
    readout_amplitude: Optional[float] = None
    """Readout pulse amplitude (optional). Same for all qubits."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ResonatorQubitSpectroscopyResults(Results):
    """ResonatorQubitSpectroscopy outputs."""

    pass


ResonatorQubitSpectroscopyType = np.dtype(
    [
        ("resonator_freq", np.float64),
        ("qubit_freq", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator qubit spectroscopy."""


@dataclass
class ResonatorQubitSpectroscopyData(Data):
    """ResonatorQubitSpectroscopy acquisition outputs."""

    data: dict[QubitId, npt.NDArray[ResonatorQubitSpectroscopyType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, resonator_freq, qubit_freq, msr, phase):
        """Store output for single qubit."""
        size = len(resonator_freq) * len(qubit_freq)
        r_freq, q_freq = np.meshgrid(resonator_freq, qubit_freq)
        ar = np.empty(size, dtype=ResonatorQubitSpectroscopyType)
        ar["resonator_freq"] = r_freq.ravel()
        ar["qubit_freq"] = q_freq.ravel()
        ar["msr"] = msr.ravel()
        ar["phase"] = phase.ravel()
        self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: ResonatorQubitSpectroscopyParameters, platform: Platform, qubits: Qubits
) -> ResonatorQubitSpectroscopyData:
    """Data acquisition for ResonatorQubit spectroscopy."""
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = []
        qd_pulse_start = 0
        for _ in range(10):
            qd_pulse = platform.create_qubit_drive_pulse(
                qubit, start=qd_pulse_start, duration=params.drive_duration
            )
            if params.drive_amplitude is not None:
                qd_pulse.amplitude = params.drive_amplitude
            qd_pulses[qubit] += [qd_pulse]
            qd_pulse_start += params.drive_duration
            sequence.add(qd_pulse)

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulse.finish
        )
        if params.readout_amplitude is not None:
            ro_pulses[qubit].amplitude = params.readout_amplitude
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    drive_delta_frequency_range = np.arange(
        -params.drive_freq_width // 2,
        params.drive_freq_width // 2,
        params.drive_freq_step,
    )
    readout_delta_frequency_range = np.arange(
        -params.readout_freq_width // 2,
        params.readout_freq_width // 2,
        params.readout_freq_step,
    )
    qubit_freq_sweeper = Sweeper(
        Parameter.frequency,
        drive_delta_frequency_range,
        pulses=[p for qubit in qubits for p in qd_pulses[qubit]],
        type=SweeperType.OFFSET,
    )
    resonator_freq_sweeper = Sweeper(
        Parameter.frequency,
        readout_delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    # Create data structure for data acquisition.
    data = ResonatorQubitSpectroscopyData()

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        qubit_freq_sweeper,
        resonator_freq_sweeper,
    )

    # retrieve the results for every qubit
    for qubit, ro_pulse in ro_pulses.items():
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulse.serial]
        # store the results
        data.register_qubit(
            qubit,
            msr=result.magnitude,
            phase=result.phase,
            resonator_freq=readout_delta_frequency_range + ro_pulses[qubit].frequency,
            qubit_freq=drive_delta_frequency_range + qd_pulses[qubit][0].frequency,
        )
    return data


def _fit(data: ResonatorQubitSpectroscopyData) -> ResonatorQubitSpectroscopyResults:
    """Post-processing function for ResonatorQubitSpectroscopy."""
    return ResonatorQubitSpectroscopyResults()


def _plot(
    data: ResonatorQubitSpectroscopyData, fit: ResonatorQubitSpectroscopyResults, qubit
):
    """Plotting function for ResonatorQubitSpectroscopy."""

    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "MSR",
            "phase (rad)",
        ),
    )

    qubit_data = data[qubit]
    resonator_frequencies = qubit_data.resonator_freq * HZ_TO_GHZ
    qubit_frequencies = qubit_data.qubit_freq * HZ_TO_GHZ

    fig.add_trace(
        go.Heatmap(
            x=resonator_frequencies,
            y=qubit_frequencies,
            z=qubit_data.msr * V_TO_UV,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text=f"{qubit}: Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Resonator Qubit Frequency", row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=resonator_frequencies,
            y=qubit_frequencies,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=f"{qubit}/: Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Resonator Qubit Frequency", row=1, col=2)

    title_text = ""

    fitting_report = fitting_report + title_text

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


resonator_qubit_spectroscopy = Routine(_acquisition, _fit, _plot)
"""ResonatorQubitSpectroscopy Routine object."""
