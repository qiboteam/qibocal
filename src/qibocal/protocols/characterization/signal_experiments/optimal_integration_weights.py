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

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine


@dataclass
class OptimalIntegrationWeightsParameters(Parameters):
    """OptimalIntegrationWeights runcard inputs."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class OptimalIntegrationWeightsResults(Results):
    """OptimalIntegrationWeights outputs."""

    optimal_integration_weights: dict[QubitId, dict[str, float]]
    """
    Optimal integration weights for a qubit given by amplifying the parts of the
    signal acquired which maximally distinguish between state 1 and 0.
    """

    def save(self, path):
        for q in self.optimal_integration_weights.keys():
            # """Helper function to use np.savez while converting keys into strings."""
            np.savez(
                path / f"optimal_integration_weights_q{q}.npz",
                self.optimal_integration_weights[q],
            )


OptimalIntegrationWeightsType = np.dtype(
    [("samples", np.complex128), ("state", np.int64)]
)


@dataclass
class OptimalIntegrationWeightsData(Data):
    """OptimalIntegrationWeights acquisition outputs."""

    data: dict[QubitId, npt.NDArray[OptimalIntegrationWeightsType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, samples, state):
        """Store output for single qubit."""
        ar = np.empty(samples.shape, dtype=OptimalIntegrationWeightsType)
        ar["samples"] = samples
        ar["state"] = state
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: OptimalIntegrationWeightsParameters, platform: Platform, qubits: Qubits
) -> OptimalIntegrationWeightsData:
    """Data acquisition for resonator spectroscopy."""

    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    # create a DataUnits object to store the results
    data = OptimalIntegrationWeightsData()

    # execute the first pulse sequence
    state0_results = platform.execute_pulse_sequence(
        state0_sequence,
        options=ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )

    # retrieve and store the results for every qubit
    for qubit in qubits:
        samples = (
            state0_results[ro_pulses[qubit].serial].voltage_i
            + 1j * state0_results[ro_pulses[qubit].serial].voltage_q
        )
        data.register_qubit(qubit, samples, [0] * len(samples))

    # execute the second pulse sequence
    state1_results = platform.execute_pulse_sequence(
        state1_sequence,
        options=ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )
    # retrieve and store the results for every qubit
    for qubit in qubits:
        samples = (
            state1_results[ro_pulses[qubit].serial].voltage_i
            + 1j * state1_results[ro_pulses[qubit].serial].voltage_q
        )
        data.register_qubit(qubit, samples, [1] * len(samples))

    return data


def _fit(data: OptimalIntegrationWeightsData) -> OptimalIntegrationWeightsResults:
    """Post-processing function for OptimalIntegrationWeights."""

    qubits = data.qubits

    # np.conj to account the two phase-space evolutions of the readout state
    optimal_integration_weights = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        truncate_index_state = np.min(np.where(qubit_data.state == 1))
        state0 = qubit_data[:truncate_index_state].samples
        state1 = qubit_data[truncate_index_state:].samples

        samples_kernel = np.conj(state1 - state0)
        # Remove nans
        samples_kernel = samples_kernel[~np.isnan(samples_kernel)]

        samples_kernel_origin = (
            samples_kernel - samples_kernel.real.min() - 1j * samples_kernel.imag.min()
        )  # origin offsetted
        samples_kernel_normalized = (
            samples_kernel_origin / np.abs(samples_kernel_origin).max()
        )  # normalized

        optimal_integration_weights[qubit] = abs(samples_kernel_normalized)

    return OptimalIntegrationWeightsResults(optimal_integration_weights)


def _plot(
    data: OptimalIntegrationWeightsData, fit: OptimalIntegrationWeightsResults, qubit
):
    """Plotting function for OptimalIntegrationWeights."""

    figures = []
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("State 0-1",),
    )

    integration_weights = fit.optimal_integration_weights[qubit]

    fitting_report = ""
    state = "1-0"

    fig.add_trace(
        go.Scatter(
            y=integration_weights,
            name=f"q{qubit}",
            showlegend=not bool(state),
            legendgroup=f"q{qubit}",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            y=np.ones([len(integration_weights)]),
            name=f"q{qubit}",
            showlegend=not bool(state),
            legendgroup=f"q{qubit}",
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Sample",
        yaxis_title="MSR (uV)",
    )

    fitting_report = fitting_report + (f"{qubit} | Optimal integration weights : <br>")

    figures.append(fig)

    return figures, fitting_report


optimal_integration_weights = Routine(_acquisition, _fit, _plot)
"""OptimalIntegrationWeights Routine object."""
