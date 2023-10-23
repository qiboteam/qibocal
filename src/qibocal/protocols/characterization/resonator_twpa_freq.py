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
class ResonatorTWPAFrequencyParameters(Parameters):
    """ResonatorTWPAFrequency runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    min_twpa_freq: int
    """TPWA frequency minimum value (Hz)."""
    max_twpa_freq: int
    """TPWA frequency maximum value (Hz)."""
    step_twpa_freq: int
    """TPWA frequency step (Hz)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ResonatorTWPAFrequencyResults(Results):
    """ResonatorTWPAFrequency outputs."""

    pass


ResonatorTWPAFrequencyType = np.dtype(
    [
        ("freq", np.float64),
        ("twpa_freq", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for Resonator TWPA Frequency."""


@dataclass
class ResonatorTWPAFrequencyData(Data):
    """ResonatorTWPAFrequency data acquisition."""

    resonator_type: str
    """Resonator type."""
    data: dict[QubitId, npt.NDArray[ResonatorTWPAFrequencyType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, twpa_freq, msr, phase):
        """Store output for single qubit."""
        size = len(freq)
        ar = np.empty(size, dtype=ResonatorTWPAFrequencyType)
        ar["freq"] = freq
        ar["twpa_freq"] = np.array([twpa_freq] * size)
        ar["msr"] = msr
        ar["phase"] = phase
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: ResonatorTWPAFrequencyParameters,
    platform: Platform,
    qubits: Qubits,
) -> ResonatorTWPAFrequencyData:
    """Data acquisition for  over TWPAFrequency."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()

    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    data = ResonatorTWPAFrequencyData(platform.resonator_type)

    # define the parameters to sweep and their range:
    # resonator frequency
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    # TWPAFrequency
    TWPAFrequency_range = np.arange(
        params.min_twpa_freq, params.max_twpa_freq, params.step_twpa_freq
    )

    for _freq in TWPAFrequency_range:
        for z in qubits:
            qubits[z].twpa.local_oscillator.frequency = _freq

        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            freq_sweeper,
        )

        # retrieve the results for every qubit
        for qubit in qubits:
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(
                qubit,
                msr=result.magnitude,
                phase=result.phase,
                freq=delta_frequency_range + ro_pulses[qubit].frequency,
                twpa_freq=_freq,
            )

    return data


def _fit(
    data: ResonatorTWPAFrequencyData, fit_type="att"
) -> ResonatorTWPAFrequencyResults:
    """Fit frequency and TWPAFrequency at high and low frequency for a given resonator."""
    return ResonatorTWPAFrequencyResults()


def _plot(
    data: ResonatorTWPAFrequencyData,
    fit: ResonatorTWPAFrequencyResults,
    qubit,
):
    """Plotting for ResonatorTWPAFrequency."""

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
    resonator_frequencies = qubit_data.freq * HZ_TO_GHZ
    twpa_frequencies = qubit_data.twpa_freq

    fig.add_trace(
        go.Heatmap(
            x=resonator_frequencies,
            y=twpa_frequencies,
            z=qubit_data.msr * V_TO_UV,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text=f"{qubit}: Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="TWPA Frequency", row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=resonator_frequencies,
            y=twpa_frequencies,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=f"{qubit}/: Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="TWPA Frequency", row=1, col=2)

    title_text = ""

    fitting_report = fitting_report + title_text

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


resonator_twpa_freq = Routine(_acquisition, _fit, _plot)
"""Resonator TWPA Frequency Routine object."""
