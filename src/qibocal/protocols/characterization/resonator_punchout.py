from dataclasses import dataclass, field
from statistics import mode
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
from qibocal.config import log

from .utils import GHZ_TO_HZ, HZ_TO_GHZ, V_TO_UV, norm


@dataclass
class ResonatorPunchoutParameters(Parameters):
    """ "ResonatorPunchout runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    amplitude: float = None
    """Initial readout amplitude."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ResonatorPunchoutResults(Results):
    """ResonatorPunchout outputs."""

    readout_frequency: dict[QubitId, float] = field(
        metadata=dict(update="readout_frequency")
    )
    """Readout frequency [GHz] for each qubit."""
    readout_amplitude: dict[QubitId, float] = field(
        metadata=dict(update="readout_amplitude")
    )
    """Readout amplitude for each qubit."""
    bare_frequency: Optional[dict[QubitId, float]] = field(
        metadata=dict(update="bare_resonator_frequency")
    )
    """Bare resonator frequency [GHz] for each qubit."""


ResPunchoutType = np.dtype(
    [
        ("freq", np.float64),
        ("amp", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator punchout."""


@dataclass
class ResonatorPunchoutData(Data):
    """ResonatorPunchout data acquisition."""

    resonator_type: str
    """Resonator type."""
    amplitudes: dict[QubitId, float]
    """Amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray[ResPunchoutType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, amp, msr, phase):
        """Store output for single qubit."""
        size = len(freq) * len(amp)
        frequency, amplitude = np.meshgrid(freq, amp)
        ar = np.empty(size, dtype=ResPunchoutType)
        ar["freq"] = frequency.ravel()
        ar["amp"] = amplitude.ravel()
        ar["msr"] = msr.ravel()
        ar["phase"] = phase.ravel()
        self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: ResonatorPunchoutParameters,
    platform: Platform,
    qubits: Qubits,
) -> ResonatorPunchoutData:
    """Data acquisition for Punchout over amplitude."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()

    ro_pulses = {}
    amplitudes = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        if params.amplitude is not None:
            ro_pulses[qubit].amplitude = params.amplitude

        amplitudes[qubit] = ro_pulses[qubit].amplitude
        sequence.add(ro_pulses[qubit])

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

    # amplitude
    amplitude_range = np.arange(
        params.min_amp_factor, params.max_amp_factor, params.step_amp_factor
    )
    amp_sweeper = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        [ro_pulses[qubit] for qubit in qubits],
        type=SweeperType.FACTOR,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and attenuation
    data = ResonatorPunchoutData(
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
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulse.serial]
        data.register_qubit(
            qubit,
            msr=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + ro_pulse.frequency,
            amp=amplitude_range * amplitudes[qubit],
        )

    return data


def _fit(data: ResonatorPunchoutData, fit_type="amp") -> ResonatorPunchoutResults:
    """Fit frequency and attenuation at high and low power for a given resonator."""

    qubits = data.qubits

    bare_freqs = {}
    dressed_freqs = {}
    ro_amplitudes = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        try:
            n_amps = len(np.unique(qubit_data.amp))
            n_freq = len(np.unique(qubit_data.freq))
            for i in range(n_amps):
                qubit_data.msr[i * n_freq : (i + 1) * n_freq] = norm(
                    qubit_data.msr[i * n_freq : (i + 1) * n_freq]
                )

            min_msr_indices = np.where(
                qubit_data.msr == (1 if data.resonator_type == "3D" else 0)
            )[0]

            max_freq = np.max(qubit_data.freq[min_msr_indices])
            min_freq = np.min(qubit_data.freq[min_msr_indices])
            middle_freq = (max_freq + min_freq) / 2

            hp_points_indices = np.where(
                qubit_data.freq[min_msr_indices] < middle_freq
            )[0]
            lp_points_indices = np.where(
                qubit_data.freq[min_msr_indices] >= middle_freq
            )[0]

            freq_hp = mode(qubit_data.freq[hp_points_indices])
            freq_lp = mode(qubit_data.freq[lp_points_indices])

            lp_max = np.max(
                getattr(qubit_data, fit_type)[np.where(qubit_data.freq == freq_lp)[0]]
            )

            ro_amp = lp_max

        except:
            log.warning("resonator_punchout_fit: the fitting was not succesful")
            freq_lp = 0.0
            freq_hp = 0.0
            ro_amp = 0.0

        dressed_freqs[qubit] = freq_lp * HZ_TO_GHZ
        bare_freqs[qubit] = freq_hp * HZ_TO_GHZ
        ro_amplitudes[qubit] = ro_amp

    return ResonatorPunchoutResults(
        dressed_freqs,
        ro_amplitudes,
        bare_freqs,
    )


def _plot(data: ResonatorPunchoutData, qubit, fit: ResonatorPunchoutResults = None):
    """Plotting function for ResonatorPunchout."""
    figures = []
    fitting_report = None

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Normalised MSR",
            "phase (rad)",
        ),
    )

    qubit_data = data[qubit]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    amplitudes = qubit_data.amp
    n_amps = len(np.unique(qubit_data.amp))
    n_freq = len(np.unique(qubit_data.freq))
    for i in range(n_amps):
        qubit_data.msr[i * n_freq : (i + 1) * n_freq] = norm(
            qubit_data.msr[i * n_freq : (i + 1) * n_freq]
        )
    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=amplitudes,
            z=qubit_data.msr * V_TO_UV,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Frequency (GHz)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (GHz)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=2)
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

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=[
                    fit.readout_frequency[qubit],
                ],
                y=[
                    fit.readout_amplitude[qubit],
                ],
                mode="markers",
                marker=dict(
                    size=8,
                    color="gray",
                    symbol="circle",
                ),
            )
        )
        title_text = ""
        title_text += f"{qubit} | Resonator frequency at low power:  {fit.readout_frequency[qubit]*GHZ_TO_HZ:,.0f} Hz<br>"
        title_text += f"{qubit} | Resonator frequency at high power: {fit.bare_frequency[qubit]*GHZ_TO_HZ:,.0f} Hz<br>"
        title_text += f"{qubit} | Readout amplitude at low power: {fit.readout_amplitude[qubit]:,.3f} <br>"

        fitting_report = fitting_report + title_text

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


resonator_punchout = Routine(_acquisition, _fit, _plot)
"""ResonatorPunchout Routine object."""
