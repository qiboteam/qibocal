from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.protocols.characterization.utils import HZ_TO_GHZ, cumulative


@dataclass
class ResonatorFrequencyParameters(Parameters):
    """Optimization RO frequency inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ResonatorFrequencyResults(Results):
    """ "Optimization RO frequency outputs."""

    fidelities: dict[QubitId, list]

    best_freq: dict[QubitId, float] = field(metadata=dict(update="readout_frequency"))


ResonatorFrequencyType = np.dtype(
    [
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
    ]
)
"""Custom dtype for Optimization RO frequency."""


@dataclass
class ResonatorFrequencyData(Data):
    """ "Optimization RO frequency acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[tuple[QubitId, int, int], npt.NDArray[ResonatorFrequencyType]] = field(
        default_factory=dict
    )

    def register_qubit(self, qubit, state, freq, msr, phase, i, q):
        """Store output for single qubit."""
        ar = np.empty(i.shape, dtype=ResonatorFrequencyType)
        ar["freq"] = freq
        ar["msr"] = msr
        ar["phase"] = phase
        self.data[qubit, state] = np.rec.array(ar)


def _acquisition(
    params: ResonatorFrequencyParameters, platform: Platform, qubits: Qubits
) -> ResonatorFrequencyData:
    r"""
    Data acquisition for readout frequency optimization.
    While sweeping the readout frequency, the routine performs a single shot
    classification and evaluates the assignement fidelity.
    At the end, the readout frequency is updated, choosing the one that has
    the highest assignment fidelity.

    Args:
        params (ResonatorFrequencyParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (dict): list of target qubits to perform the action

    """

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence_0.add(ro_pulses[qubit])
        sequence_1.add(qd_pulses[qubit])
        sequence_1.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    data = ResonatorFrequencyData(platform.resonator_type)
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    results_0 = platform.sweep(
        sequence_0,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        ),
        sweeper,
    )

    results_1 = platform.sweep(
        sequence_1,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        ),
        sweeper,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        for i, results in enumerate([results_0, results_1]):
            result = results[ro_pulses[qubit].serial]
            # store the results
            data.register_qubit(
                qubit=qubit,
                state=i,
                freq=ro_pulses[qubit].frequency + delta_frequency_range,
                i=result.voltage_i,
                q=result.voltage_q,
            )
    return data


def _fit(data: ResonatorFrequencyData) -> ResonatorFrequencyResults:
    """Post-Processing for Optimization RO frequency"""
    qubits = data.qubits
    fidelities_dict = {}
    best_freqs = {}
    for qubit in qubits:
        fidelities = []
        freqs = np.unique(data[qubit, 0].freq)
        for freq in freqs:
            iq_state0 = data[qubit, 0][data[qubit, 0].freq == freq][["i", "q"]]
            iq_state1 = data[qubit, 1][data[qubit, 1].freq == freq][["i", "q"]]
            iq_state0 = iq_state0.i + 1.0j * iq_state0.q
            iq_state1 = iq_state1.i + 1.0j * iq_state1.q

            iq_state1 = np.array(iq_state1)
            iq_state0 = np.array(iq_state0)
            nshots = len(iq_state0)

            iq_mean_state1 = np.mean(iq_state1)
            iq_mean_state0 = np.mean(iq_state0)

            vector01 = iq_mean_state1 - iq_mean_state0
            rotation_angle = np.angle(vector01)

            iq_state1_rotated = iq_state1 * np.exp(-1j * rotation_angle)
            iq_state0_rotated = iq_state0 * np.exp(-1j * rotation_angle)

            real_values_state1 = iq_state1_rotated.real
            real_values_state0 = iq_state0_rotated.real

            real_values_combined = np.concatenate(
                (real_values_state1, real_values_state0)
            )
            real_values_combined.sort()

            cum_distribution_state1 = cumulative(
                real_values_combined, real_values_state1
            )
            cum_distribution_state0 = cumulative(
                real_values_combined, real_values_state0
            )

            cum_distribution_diff = np.abs(
                np.array(cum_distribution_state1) - np.array(cum_distribution_state0)
            )
            argmax = np.argmax(cum_distribution_diff)
            errors_state1 = nshots - cum_distribution_state1[argmax]
            errors_state0 = cum_distribution_state0[argmax]
            fidelities.append((errors_state1 + errors_state0) / nshots / 2)
        fidelities_dict[qubit] = fidelities
        best_freqs[qubit] = freqs[np.argmax(fidelities_dict[qubit])]

    return ResonatorFrequencyResults(
        fidelities=fidelities_dict,
        best_freq=best_freqs,
    )


def _plot(data: ResonatorFrequencyData, fit: ResonatorFrequencyResults, qubit):
    """Plotting function for Optimization RO frequency."""
    figures = []
    freqs = np.unique(data[qubit, 0].freq) * HZ_TO_GHZ
    opacity = 1
    fitting_report = " "
    fig = make_subplots(
        rows=1,
        cols=1,
    )

    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=fit.fidelities[qubit],
            opacity=opacity,
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Resonator Frequencies (GHz)",
        yaxis_title="Assignment Fidelities",
    )

    fitting_report = fitting_report + (
        f"{qubit} | Best Resonator Frequency (GHz) : {fit.best_freq[qubit]*HZ_TO_GHZ:,.4f} Hz.<br>"
    )

    figures.append(fig)

    return figures, fitting_report


resonator_frequency = Routine(_acquisition, _fit, _plot)
""""Optimization RO frequency Routine object."""
