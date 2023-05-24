from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from . import utils


# TODO: implement cross-talk (maybe separate routine?)
@dataclass
class ResonatorFluxParameters(Parameters):
    """ResonatorFlux runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [V]."""
    bias_step: float
    """Bias step for sweep [V]."""
    nshots: int
    """Number of shots."""
    relaxation_time: int
    """Relaxation time [ns]."""


@dataclass
class ResonatorFluxResults(Results):
    """ResonatoFlux outputs."""

    sweetspot: Dict[Union[str, int], float] = field(metadata=dict(update="sweetspot"))
    """Sweetspot for each qubit."""
    frequency: Dict[Union[str, int], float] = field(
        metadata=dict(update="readout_frequency")
    )
    """Readout frequency for each qubit."""
    fitted_parameters: Dict[Union[str, int], Dict[str, float]]
    """Raw fitting output."""


class ResonatorFluxData(DataUnits):
    """ResonatorFlux acquisition outputs."""

    def __init__(self):
        super().__init__(
            "data",
            {"frequency": "Hz", "bias": "V"},
            options=["qubit"],
        )


def _acquisition(
    params: ResonatorFluxParameters, platform: AbstractPlatform, qubits: Qubits
) -> ResonatorFluxData:
    """Data acquisition for ResonatorFlux experiment."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    bias_sweeper = Sweeper(
        Parameter.bias, delta_bias_range, qubits=list(qubits.values())
    )
    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and flux bias
    data = ResonatorFluxData()

    # repeat the experiment as many times as defined by software_averages
    results = platform.sweep(
        sequence,
        bias_sweeper,
        freq_sweeper,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]

        biases = np.repeat(delta_bias_range, len(delta_frequency_range))
        freqs = np.array(
            len(delta_bias_range)
            * list(delta_frequency_range + ro_pulses[qubit].frequency)
        ).flatten()
        # store the results
        r = {k: v.ravel() for k, v in result.raw.items()}
        r.update(
            {
                "frequency[Hz]": freqs,
                "bias[V]": biases,
                "qubit": len(freqs) * [qubit],
            }
        )
        data.add_data_from_dict(r)

    return data


def _fit(data: ResonatorFluxData) -> ResonatorFluxResults:
    """Post-processing for ResonatorFlux experiment."""
    return ResonatorFluxResults({}, {}, {})


def _plot(data: ResonatorFluxData, fit: ResonatorFluxResults, qubit):
    """Plotting function for ResonatorFlux Experiment."""
    return utils.flux_dependence_plot(data, fit, qubit)


resonator_flux = Routine(_acquisition, _fit, _plot)
"""ResonatorFlux Routine object."""
