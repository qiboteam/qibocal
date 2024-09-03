from dataclasses import dataclass, field

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
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import table_dict, table_html


@dataclass
class ResonatorFrequencyParameters(Parameters):
    """Optimization RO frequency inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""


@dataclass
class ResonatorFrequencyResults(Results):
    """Optimization Resonator frequency outputs."""

    fidelities: dict[QubitId, list]
    """Assignment fidelities."""
    best_freq: dict[QubitId, float]
    """Resonator Frequency with the highest assignment fidelity."""
    best_angle: dict[QubitId, float]
    """IQ angle that maximes assignment fidelity"""
    best_threshold: dict[QubitId, float]
    """Threshold that maximes assignment fidelity"""


ResonatorFrequencyType = np.dtype(
    [
        ("freq", np.float64),
        ("assignment_fidelity", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
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

    def unique_freqs(self, qubit: QubitId) -> np.ndarray:
        return np.unique(self.data[qubit]["freq"])


def _acquisition(
    params: ResonatorFrequencyParameters, platform: Platform, targets: list[QubitId]
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
        qubits (list): list of target qubits to perform the action

    """

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in targets:
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
        pulses=[ro_pulses[qubit] for qubit in targets],
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
    for qubit in targets:
        for k, freq in enumerate(delta_frequency_range):
            i_values = []
            q_values = []
            states = []
            for i, results in enumerate([results_0, results_1]):
                result = results[ro_pulses[qubit].serial]
                i_values.extend(result.voltage_i[k])
                q_values.extend(result.voltage_q[k])
                states.extend([i] * len(result.voltage_i[k]))

            model = QubitFit()
            model.fit(np.stack((i_values, q_values), axis=-1), np.array(states))
            data.register_qubit(
                ResonatorFrequencyType,
                (qubit),
                dict(
                    freq=np.array([(ro_pulses[qubit].frequency + freq)]),
                    assignment_fidelity=np.array([model.assignment_fidelity]),
                    angle=np.array([model.angle]),
                    threshold=np.array([model.threshold]),
                ),
            )
    return data


def _fit(data: ResonatorFrequencyData) -> ResonatorFrequencyResults:
    """Post-Processing for Optimization RO frequency"""
    qubits = data.qubits
    best_freq = {}
    best_angle = {}
    best_threshold = {}
    highest_fidelity = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_fid = np.argmax(data_qubit["assignment_fidelity"])
        highest_fidelity[qubit] = data_qubit["assignment_fidelity"][index_best_fid]
        best_freq[qubit] = data_qubit["freq"][index_best_fid]
        best_angle[qubit] = data_qubit["angle"][index_best_fid]
        best_threshold[qubit] = data_qubit["threshold"][index_best_fid]

    return ResonatorFrequencyResults(
        fidelities=highest_fidelity,
        best_freq=best_freq,
        best_angle=best_angle,
        best_threshold=best_threshold,
    )


def _plot(
    data: ResonatorFrequencyData, fit: ResonatorFrequencyResults, target: QubitId
):
    """Plotting function for Optimization RO frequency."""
    figures = []
    freqs = data[target]["freq"]
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
                y=data[target]["assignment_fidelity"],
                opacity=opacity,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                target,
                "Best Resonator Frequency [Hz]",
                np.round(fit.best_freq[target], 4),
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Resonator Frequencies [GHz]",
        yaxis_title="Assignment Fidelities",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ResonatorFrequencyResults, platform: Platform, target: QubitId):
    update.readout_frequency(results.best_freq[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)


resonator_frequency = Routine(_acquisition, _fit, _plot, _update)
""""Optimization RO frequency Routine object."""
