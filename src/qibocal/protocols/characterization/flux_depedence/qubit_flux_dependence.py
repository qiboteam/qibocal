from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine

from . import resonator_flux_dependence, utils


# TODO: implement cross-talk
@dataclass
class QubitFluxParameters(Parameters):
    """QubitFlux runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the qubit frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [V]."""
    bias_step: float
    """Bias step for sweep (V)."""
    drive_amplitude: float
    """Drive pulse amplitude. Same for all qubits."""
    qubits: list = field(default_factory=list)
    """Local qubits (optional)."""
    update: Optional[bool] = None
    """Runcard update mechanism."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class QubitFluxResults(Results):
    """QubitFlux outputs."""

    sweetspot: Dict[QubitId, float] = field(metadata=dict(update="sweetspot"))
    """Sweetspot for each qubit."""
    frequency: Dict[QubitId, float] = field(metadata=dict(update="drive_frequency"))
    """Drive frequency for each qubit."""
    fitted_parameters: Dict[QubitId, Dict[str, float]]
    """Raw fitting output."""


class QubitFluxData(resonator_flux_dependence.ResonatorFluxData):
    """QubitFlux acquisition outputs."""


def _acquisition(
    params: QubitFluxParameters,
    platform: Platform,
    qubits: Qubits,
) -> QubitFluxData:
    """Data acquisition for QubitFlux Experiment."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=2000
        )
        qd_pulses[qubit].amplitude = params.drive_amplitude
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
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
    data = QubitFluxData()

    # repeat the experiment as many times as defined by software_averages
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        bias_sweeper,
        freq_sweeper,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]

        biases = np.repeat(delta_bias_range, len(delta_frequency_range))
        freqs = np.array(
            len(delta_bias_range)
            * list(delta_frequency_range + qd_pulses[qubit].frequency)
        ).flatten()
        # store the results
        r = {k: v.ravel() for k, v in result.serialize.items()}
        r.update(
            {
                "frequency[Hz]": freqs,
                "bias[V]": biases,
                "qubit": len(freqs) * [qubit],
            }
        )
        data.add_data_from_dict(r)

    return data


def _fit(data: QubitFluxData) -> QubitFluxResults:
    """Post-processing for QubitFlux Experiment."""
    return QubitFluxResults({}, {}, {})


def _plot(data: QubitFluxData, fit: QubitFluxResults, qubit):
    """Plotting function for QubitFlux Experiment."""
    return utils.flux_dependence_plot(data, fit, qubit)


qubit_flux = Routine(_acquisition, _fit, _plot)
"""QubitFlux Routine object."""
