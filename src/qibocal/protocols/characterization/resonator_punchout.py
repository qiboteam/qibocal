from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

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

from . import utils


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

    readout_frequency: Dict[QubitId, float] = field(
        metadata=dict(update="readout_frequency")
    )
    """Readout frequency [GHz] for each qubit."""
    readout_amplitude: Dict[QubitId, float] = field(
        metadata=dict(update="readout_amplitude")
    )
    """Readout amplitude for each qubit."""
    bare_frequency: Optional[Dict[QubitId, float]] = field(
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
    amplitudes: Dict[QubitId, float]
    """Amplitudes provided by the user."""
    data: Dict[QubitId, npt.NDArray[ResPunchoutType]] = field(default_factory=dict)
    """Raw data acquired."""

    def add_qubit_data(self, qubit, freq, amp, msr, phase):
        """Store output for single qubit."""
        ar = np.empty(msr.shape, dtype=ResPunchoutType)
        frequency, amplitude = np.meshgrid(freq, amp)
        ar["freq"] = frequency.flatten()
        ar["amp"] = amplitude.flatten()
        ar["msr"] = msr
        ar["phase"] = phase
        self.data[qubit] = np.rec.array(ar)

    @property
    def qubits(self):
        """Access qubits from data structure."""
        return [q for q in self.data]

    @property
    def global_params_dict(self):
        global_dict = asdict(self)
        global_dict.pop("data")
        return global_dict

    def __getitem__(self, qubit):
        return self.data[qubit]

    def save(self, path):
        """Store results."""
        self.to_json(path, self.global_params_dict)
        self.to_npz(path, self.data)


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

    # repeat the experiment as many times as defined by software_averages
    amps = np.repeat(amplitude_range, len(delta_frequency_range))

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
        data.add_qubit_data(
            qubit,
            msr=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + ro_pulse.frequency,
            amp=amplitude_range,
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
                qubit_data.msr[i * n_freq : (i + 1) * n_freq] = utils.norm(
                    qubit_data.msr[i * n_freq : (i + 1) * n_freq]
                )

            # averaged_data_updated = qubit_data.copy()
            # averaged_data_updated.update(normalised_data["MSR"])

            min_points = utils.find_min_msr(qubit_data, data.resonator_type, fit_type)

            max_freq = np.max(qubit_data.freq[min_points])
            min_freq = np.min(qubit_data.freq[min_points])
            middle_freq = (max_freq + min_freq) / 2

            hp_points_indices = np.where(qubit_data.freq[min_points] < middle_freq)[0]
            lp_points_indices = np.where(qubit_data.freq[min_points] >= middle_freq)[0]

            # TODO: remove this function
            freq_hp = utils.get_max_freq(qubit_data.freq[hp_points_indices])
            freq_lp = utils.get_max_freq(qubit_data.freq[lp_points_indices])

            # TODO: implement if else in previous function
            lp_max = np.max(
                getattr(qubit_data, fit_type)[np.where(qubit_data.freq == freq_lp)[0]]
            )
            hp_max = np.max(
                getattr(qubit_data, fit_type)[np.where(qubit_data.freq == freq_hp)[0]]
            )

            ro_amp = lp_max

        except:
            log.warning("resonator_punchout_fit: the fitting was not succesful")
            freq_lp = 0.0
            freq_hp = 0.0
            ro_amp = 0.0

        dressed_freqs[qubit] = freq_lp / 1e9
        bare_freqs[qubit] = freq_hp / 1e9
        ro_amplitudes[qubit] = ro_amp

    return ResonatorPunchoutResults(
        dressed_freqs,
        ro_amplitudes,
        bare_freqs,
    )


def _plot(data: ResonatorPunchoutData, fit: ResonatorPunchoutResults, qubit):
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
    frequencies = qubit_data.freq / 1e9
    amplitudes = qubit_data.amp
    n_amps = len(np.unique(qubit_data.amp))
    n_freq = len(np.unique(qubit_data.freq))
    for i in range(n_amps):
        qubit_data.msr[i * n_freq : (i + 1) * n_freq] = utils.norm(
            qubit_data.msr[i * n_freq : (i + 1) * n_freq]
        )

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=amplitudes,
            z=qubit_data.msr * 1e4,
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
    title_text += f"{qubit} | Resonator frequency at low power:  {fit.readout_frequency[qubit]*1e9:,.0f} Hz<br>"
    title_text += f"{qubit} | Resonator frequency at high power: {fit.bare_frequency[qubit]*1e9:,.0f} Hz<br>"
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
