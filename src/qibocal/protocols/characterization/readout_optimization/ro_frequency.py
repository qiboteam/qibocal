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

from ..utils import HZ_TO_GHZ, cumulative


@dataclass
class RoFrequencyParameters(Parameters):
    """Dispersive shift inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class RoFrequencyResults(Results):
    """Dispersive shift outputs."""

    fidelities: dict[QubitId, list]

    best_freq: dict[QubitId, float] = field(metadata=dict(update="readout_frequency"))


RoFrequencyType = np.dtype(
    [
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RoFrequencyData(Data):
    """Dipsersive shift acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[tuple[QubitId, int, int], npt.NDArray[RoFrequencyType]] = field(
        default_factory=dict
    )

    def register_qubit(self, qubit, state, freq, msr, phase, i, q):
        """Store output for single qubit."""
        ar = np.empty(i.shape, dtype=RoFrequencyType)
        ar["freq"] = freq
        ar["msr"] = msr
        ar["phase"] = phase
        ar["i"] = i
        ar["q"] = q
        self.data[qubit, state] = np.rec.array(ar)


def _acquisition(
    params: RoFrequencyParameters, platform: Platform, qubits: Qubits
) -> RoFrequencyData:
    r"""
    Data acquisition for dispersive shift experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (RoFrequencyParameters): experiment's parameters
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

    data = RoFrequencyData(platform.resonator_type)
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
            # averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweeper,
    )

    results_1 = platform.sweep(
        sequence_1,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            # averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweeper,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        for i, results in enumerate([results_0, results_1]):
            result = results[ro_pulses[qubit].serial]
            # store the results
            data.register_qubit(
                qubit=qubit,
                state=i,
                freq=ro_pulses[qubit].frequency + delta_frequency_range,
                msr=result.magnitude,
                phase=result.phase,
                i=result.voltage_i,
                q=result.voltage_q,
            )
    return data


def _fit(data: RoFrequencyData) -> RoFrequencyResults:
    """Post-Processing for dispersive shift"""
    qubits = data.qubits
    freqs = np.unique(data[qubits[0], 0].freq)
    fidelities_dict = {}
    fidelities = []
    best_freqs = {}
    for qubit in qubits:
        for freq in freqs:
            iq_state0 = data[qubit, 0][data[qubit, 0].freq == freq][["i", "q"]]
            iq_state1 = data[qubit, 1][data[qubit, 1].freq == freq][["i", "q"]]

            import matplotlib.pyplot as plt

            plt.subplots()
            plt.scatter(iq_state0.i, iq_state0.q)
            plt.scatter(iq_state1.i, iq_state1.q)

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

    return RoFrequencyResults(
        fidelities=fidelities_dict,
        best_freq=best_freqs,
    )


def _plot(data: RoFrequencyData, fit: RoFrequencyResults, qubit):
    """Plotting function for dispersive shift."""
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
        xaxis_title="RO Frequencies (GHz)",
        yaxis_title="Assignment Fidelities",
    )

    fitting_report = fitting_report + (
        f"{qubit} | Best RO Frequency (GHz) : {fit.best_freq[qubit]*HZ_TO_GHZ:,.4f} Hz.<br>"
    )

    figures.append(fig)

    return figures, fitting_report


ro_frequency = Routine(_acquisition, _fit, _plot)
"""Dispersive shift Routine object."""