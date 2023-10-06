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

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.characterization.utils import HZ_TO_GHZ, table_dict, table_html


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
    """Optimization Resonator frequency outputs."""

    fidelities: dict[QubitId, list]
    """Assignment fidelities."""
    best_freq: dict[QubitId, float]
    """Resonator Frequency with the highest assignment fidelity."""


ResonatorFrequencyType = np.dtype(
    [
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
        ("state", int),
    ]
)
"""Custom dtype for Optimization RO frequency."""


@dataclass
class ResonatorFrequencyData(Data):
    """ "Optimization RO frequency acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[QubitId, npt.NDArray[ResonatorFrequencyType]] = field(
        default_factory=dict
    )

    def append_data(self, qubit, state, freq, i, q):
        """Append elements to data for single qubit."""
        ar = np.empty(i.shape, dtype=ResonatorFrequencyType)
        ar["freq"] = freq
        ar["i"] = i
        ar["q"] = q
        ar["state"] = state
        if qubit in self.data.keys():
            self.data[qubit] = np.append(self.data[qubit], np.rec.array(ar))
        else:
            self.data[qubit] = np.rec.array(ar)

    def unique_freqs(self, qubit: QubitId) -> np.ndarray:
        return np.unique(self.data[qubit]["freq"])


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

    data = ResonatorFrequencyData(resonator_type=platform.resonator_type)
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
            data.append_data(
                qubit=qubit,
                state=i,
                freq=(ro_pulses[qubit].frequency + delta_frequency_range) * HZ_TO_GHZ,
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
        freqs = data.unique_freqs(qubit)
        for freq in freqs:
            fit_method = QubitFit()
            data_freq = data[qubit][data[qubit]["freq"] == freq]
            iq_couples = np.array(data_freq[["i", "q"]].tolist())[:, :]
            states = np.array(data_freq[["state"]].tolist())[:, 0]
            fit_method.fit(iq_couples, states)
            fidelities.append(fit_method.assignment_fidelity)
        fidelities_dict[qubit] = fidelities
        best_freqs[qubit] = freqs[np.argmax(fidelities_dict[qubit])]

    return ResonatorFrequencyResults(
        fidelities=fidelities_dict,
        best_freq=best_freqs,
    )


def _plot(data: ResonatorFrequencyData, fit: ResonatorFrequencyResults, qubit):
    """Plotting function for Optimization RO frequency."""
    figures = []
    freqs = data.unique_freqs(qubit)
    opacity = 1
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=1,
    )

    if fit is not None:
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

        fitting_report = table_html(
            table_dict(
                qubit,
                "Best Resonator Frequency [GHz]",
                fit.best_freq[qubit] * HZ_TO_GHZ,
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Resonator Frequencies (GHz)",
        yaxis_title="Assignment Fidelities",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ResonatorFrequencyResults, platform: Platform, qubit: QubitId):
    update.readout_frequency(results.best_freq[qubit], platform, qubit)


resonator_frequency = Routine(_acquisition, _fit, _plot, _update)
""""Optimization RO frequency Routine object."""
