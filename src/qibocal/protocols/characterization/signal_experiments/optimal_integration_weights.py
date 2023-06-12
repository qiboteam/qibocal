from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits
from qibocal.protocols.characterization.signal_experiments.utils import signal_0_1


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

    optimal_integration_weights: Dict[Union[str, int], float]
    """
    Optimal integration weights for a qubit given by amplifying the parts of the
    signal acquired which maximally distinguish between state 1 and 0.
    """


class OptimalIntegrationWeightsData(DataUnits):
    """OptimalIntegrationWeights acquisition outputs."""

    def __init__(self):
        super().__init__(
            "data",
            options=["qubit", "sample", "state"],
        )


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
    data = DataUnits(
        name="data",
        quantities={"weights": "dimensionless"},
        options=["qubit", "sample", "state"],
    )

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
        r = state0_results[ro_pulses[qubit].serial].serialize
        number_of_samples = len(r["MSR[V]"])
        r.update(
            {
                "qubit": [qubit] * number_of_samples,
                "sample": np.arange(number_of_samples),
                "state": [0] * number_of_samples,
            }
        )
        data.add_data_from_dict(r)

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
        # average msr, phase, i and q over the number of shots defined in the runcard
        r = state1_results[ro_pulses[qubit].serial].serialize
        number_of_samples = len(r["MSR[V]"])
        r.update(
            {
                "qubit": [qubit] * number_of_samples,
                "sample": np.arange(number_of_samples),
                "state": [1] * number_of_samples,
            }
        )
        data.add_data_from_dict(r)

    return data


def _fit(data: OptimalIntegrationWeightsData) -> OptimalIntegrationWeightsResults:
    """Post-processing function for OptimalIntegrationWeights."""

    qubits = data.df["qubit"].unique()

    # np.conj to account the two phase-space evolutions of the readout state
    integration_weights = {}

    for qubit in qubits:
        qubit_data_df = data.df[data.df["qubit"] == qubit]

        qubit_state0_data_df = qubit_data_df[qubit_data_df["state"] == 0]
        qubit_state1_data_df = qubit_data_df[qubit_data_df["state"] == 1]

        state0 = (
            qubit_state0_data_df["i"].pint.to("uV").pint.magnitude
            + 1j * qubit_state0_data_df["q"].pint.to("uV").pint.magnitude
        )
        state1 = (
            qubit_state1_data_df["i"].pint.to("uV").pint.magnitude
            + 1j * qubit_state1_data_df["q"].pint.to("uV").pint.magnitude
        )

        number_of_samples = len(qubit_state0_data_df["i"].pint.to("uV").pint.magnitude)

        state0 = state0.to_numpy()
        state1 = state1.to_numpy()

        samples_kernel = np.conj(state1 - state0)
        # Remove nans
        samples_kernel = samples_kernel[~np.isnan(samples_kernel)]

        samples_kernel_origin = (
            samples_kernel - samples_kernel.real.min() - 1j * samples_kernel.imag.min()
        )  # origin offsetted
        samples_kernel_normalized = (
            samples_kernel_origin / np.abs(samples_kernel_origin).max()
        )  # normalized

        integration_weights[qubit] = abs(samples_kernel_normalized)

    return OptimalIntegrationWeightsResults(integration_weights)


def _plot(
    data: OptimalIntegrationWeightsData, fit: OptimalIntegrationWeightsResults, qubit
):
    """Plotting function for OptimalIntegrationWeights."""
    return signal_0_1(data, fit, qubit)


optimal_integration_weights = Routine(_acquisition, _fit, _plot)
"""OptimalIntegrationWeights Routine object."""
