from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from .utils import signal_0_1


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
            {"frequency": "Hz"},
            options=["qubit"],
        )


def _acquisition(
    params: OptimalIntegrationWeightsParameters, platform: Platform, qubits: Qubits
) -> OptimalIntegrationWeightsData:
    """Data acquisition for resonator spectroscopy."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel

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
            r = state0_results[ro_pulses[qubit].serial].raw
            state0 = r["i[V]"] + 1j * r["q[V]"]
            number_of_samples = len(r["MSR[V]"])
            r.update(
                {
                    "qubit": [ro_pulse.qubit] * len(r["MSR[V]"]),
                    "sample": np.arange(len(r["MSR[V]"])),
                    "state": [0] * len(r["MSR[V]"]),
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
            r = state1_results[ro_pulses[qubit].serial].raw
            state1 = r["i[V]"] + 1j * r["q[V]"]
            r.update(
                {
                    "qubit": [ro_pulse.qubit] * len(r["MSR[V]"]),
                    "sample": np.arange(len(r["MSR[V]"])),
                    "state": [1] * len(r["MSR[V]"]),
                }
            )
            data.add_data_from_dict(r)

    # retrieve the results for every qubit
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        # store the results
        r = result.serialize
        r.update(
            {
                "frequency[Hz]": delta_frequency_range + ro_pulses[qubit].frequency,
                "qubit": len(delta_frequency_range) * [qubit],
            }
        )
        data.add_data_from_dict(r)
    # finally, save the remaining data
    return data


def _fit(data: OptimalIntegrationWeightsData) -> OptimalIntegrationWeightsResults:
    """Post-processing function for OptimalIntegrationWeights."""
    return OptimalIntegrationWeightsResults({})


def _plot(
    data: OptimalIntegrationWeightsData, fit: OptimalIntegrationWeightsResults, qubit
):
    """Plotting function for OptimalIntegrationWeights."""
    return signal_0_1(data, fit, qubit)


optimal_integration_weights = Routine(_acquisition, _fit, _plot)
"""OptimalIntegrationWeights Routine object."""
