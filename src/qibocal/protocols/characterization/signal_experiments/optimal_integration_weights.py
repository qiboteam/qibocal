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


OptimalIntegrationWeightsType = np.dtype([("samples", np.complex128)])


@dataclass
class OptimalIntegrationWeightsData(Data):
    """OptimalIntegrationWeights acquisition outputs."""

    data: dict[tuple[QubitId, int], npt.NDArray[OptimalIntegrationWeightsType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, samples, state):
        """Store output for single qubit."""
        ar = np.empty(samples.shape, dtype=OptimalIntegrationWeightsType)
        ar["samples"] = samples
        self.data[qubit, state] = np.rec.array(ar)


def _acquisition(
    params: OptimalIntegrationWeightsParameters, platform: Platform, qubits: Qubits
) -> OptimalIntegrationWeightsData:
    """Data acquisition for resonator spectroscopy."""

    # create a DataUnits object to store the results
    data = OptimalIntegrationWeightsData()
    for state in [0, 1]:
        if state == 1:
            RX_pulses = {}
        ro_pulses = {}
        sequence = PulseSequence()
        for qubit in qubits:
            if state == 1:
                RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
                ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                    qubit, start=RX_pulses[qubit].finish
                )
                sequence.add(RX_pulses[qubit])
            else:
                ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
                sequence.add(ro_pulses[qubit])
            sequence.add(ro_pulses[qubit])
        # execute the first pulse sequence
        results = platform.execute_pulse_sequence(
            sequence,
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
                results[ro_pulses[qubit].serial].voltage_i
                + 1j * results[ro_pulses[qubit].serial].voltage_q
            )
            data.register_qubit(qubit, samples, state)
    return data


def _fit(data: OptimalIntegrationWeightsData) -> OptimalIntegrationWeightsResults:
    """Post-processing function for OptimalIntegrationWeights."""

    qubits = data.qubits

    # np.conj to account the two phase-space evolutions of the readout state
    optimal_integration_weights = {}

    for qubit in qubits:
        state0 = data[qubit, 0].samples
        state1 = data[qubit, 1].samples

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
