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

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from .utils import (
    GHZ_TO_HZ,
    HZ_TO_GHZ,
    V_TO_UV,
    fit_punchout,
    norm,
    table_dict,
    table_html,
)


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


@dataclass
class ResonatorPunchoutResults(Results):
    """ResonatorPunchout outputs."""

    readout_frequency: dict[QubitId, float]
    """Readout frequency [GHz] for each qubit."""
    bare_frequency: Optional[dict[QubitId, float]]
    """Bare resonator frequency [GHz] for each qubit."""
    readout_amplitude: dict[QubitId, float]
    """Readout amplitude for each qubit."""


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

    return ResonatorPunchoutResults(*fit_punchout(data, fit_type))


def _plot(data: ResonatorPunchoutData, qubit, fit: ResonatorPunchoutResults = None):
    """Plotting function for ResonatorPunchout."""
    figures = []
    fitting_report = ""
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
        fitting_report = table_html(
            table_dict(
                qubit,
                [
                    "Low Power Resonator Frequency",
                    "Low Power readout amplitude",
                    "High Power Resonator Frequency",
                ],
                [
                    np.round(fit.readout_frequency[qubit] * GHZ_TO_HZ),
                    np.round(fit.readout_amplitude[qubit], 3),
                    np.round(fit.bare_frequency[qubit] * GHZ_TO_HZ),
                ],
            )
        )

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ResonatorPunchoutResults, platform: Platform, qubit: QubitId):
    update.readout_frequency(results.readout_frequency[qubit], platform, qubit)
    update.bare_resonator_frequency(results.bare_frequency[qubit], platform, qubit)
    update.readout_amplitude(results.readout_amplitude[qubit], platform, qubit)


resonator_punchout = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorPunchout Routine object."""
