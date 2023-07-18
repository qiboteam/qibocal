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

from .utils import GHZ_TO_HZ, HZ_TO_GHZ, V_TO_UV, fit_punchout, norm


@dataclass
class ResonatorPunchoutAttenuationParameters(Parameters):
    """ResonatorPunchoutAttenuation runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    min_att: int
    """Attenuation minimum value (dB)."""
    max_att: int
    """Attenuation maximum value (dB)."""
    step_att: int
    """Attenuation step (dB)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ResonatorPunchoutAttenuationResults(Results):
    """ResonatorPunchoutAttenation outputs."""

    readout_frequency: dict[QubitId, float] = field(
        metadata=dict(update="readout_frequency")
    )
    """Readout frequency [GHz] for each qubit."""
    bare_frequency: Optional[dict[QubitId, float]] = field(
        metadata=dict(update="bare_resonator_frequency")
    )
    readout_attenuation: dict[QubitId, int] = field(
        metadata=dict(update="readout_attenuation")
    )
    """Readout attenuation [dB] for each qubit."""


ResPunchoutAttType = np.dtype(
    [
        ("freq", np.float64),
        ("att", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator punchout."""


@dataclass
class ResonatorPunchoutAttenuationData(Data):
    """ResonatorPunchoutAttenuation data acquisition."""

    resonator_type: str
    """Resonator type."""
    data: dict[QubitId, npt.NDArray[ResPunchoutAttType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, att, msr, phase):
        """Store output for single qubit."""
        size = len(freq) * len(att)
        ar = np.empty(size, dtype=ResPunchoutAttType)
        frequency, attenuation = np.meshgrid(freq, att)
        ar["freq"] = frequency.ravel()
        ar["att"] = attenuation.ravel()
        ar["msr"] = msr.ravel()
        ar["phase"] = phase.ravel()
        self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: ResonatorPunchoutAttenuationParameters,
    platform: Platform,
    qubits: Qubits,
) -> ResonatorPunchoutAttenuationData:
    """Data acquisition for Punchout over attenuation."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()

    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
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

    # attenuation
    attenuation_range = np.arange(params.min_att, params.max_att, params.step_att)
    att_sweeper = Sweeper(
        Parameter.attenuation,
        attenuation_range,
        qubits=list(qubits.values()),
        type=SweeperType.ABSOLUTE,
    )

    data = ResonatorPunchoutAttenuationData(platform.resonator_type)

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        att_sweeper,
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
            att=attenuation_range,
        )

        # # Temporary fixe to force to reset the attenuation to the original value in qblox
        # # sweeper method returning to orig value not working for attenuation
        # # After fitting punchout the reload_settings will be called automatically
        # platform.reload_settings()
        # save data
    return data


def _fit(
    data: ResonatorPunchoutAttenuationData, fit_type="att"
) -> ResonatorPunchoutAttenuationResults:
    """Fit frequency and attenuation at high and low power for a given resonator."""

    return ResonatorPunchoutAttenuationResults(*fit_punchout(data, fit_type))


def _plot(
    data: ResonatorPunchoutAttenuationData,
    fit: ResonatorPunchoutAttenuationResults,
    qubit,
):
    """Plotting for ResonatorPunchoutAttenuation."""

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
    attenuations = qubit_data.att
    n_att = len(np.unique(qubit_data.att))
    n_freq = len(np.unique(qubit_data.freq))
    for i in range(n_att):
        qubit_data.msr[i * n_freq : (i + 1) * n_freq] = norm(
            qubit_data.msr[i * n_freq : (i + 1) * n_freq]
        )

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=attenuations,
            z=qubit_data.msr * V_TO_UV,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text=f"{qubit}: Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Attenuation", row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=attenuations,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=f"{qubit}/: Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Attenuation", row=1, col=2)

    fig.add_trace(
        go.Scatter(
            x=[
                fit.readout_frequency[qubit],
            ],
            y=[
                fit.readout_attenuation[qubit],
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
    title_text += f"{qubit} | Resonator Frequency at Low Power:  {fit.readout_frequency[qubit] * GHZ_TO_HZ} Hz.<br>"
    title_text += (
        f"{qubit} | Readout Attenuation: {fit.readout_attenuation[qubit]} db.<br>"
    )
    title_text += f"{qubit} | Resonator Frequency at High Power: {fit.bare_frequency[qubit] * GHZ_TO_HZ} Hz.<br>"

    fitting_report = fitting_report + title_text

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


resonator_punchout_attenuation = Routine(_acquisition, _fit, _plot)
"""ResonatorPunchoutAttenuation Routine object."""
