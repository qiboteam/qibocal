import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Qubits, Routine

from .ramsey import RamseyData, RamseyParameters, RamseyType, _fit, _plot, _update
from .utils import ramsey_sequence


def _acquisition(
    params: RamseyParameters,
    platform: Platform,
    qubits: Qubits,
) -> RamseyData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ

    # define the parameter to sweep and its range:
    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    data = RamseyData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.qubits[qubit].native_gates.RX.frequency for qubit in qubits
        },
    )

    # sweep the parameter
    for wait in waits:
        sequence = PulseSequence()
        for qubit in qubits:
            sequence += ramsey_sequence(
                platform=platform, qubit=qubit, wait=wait, detuning=params.detuning
            )
        # execute the pulse sequence
        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.SINGLESHOT,
            ),
        )

        for qubit in qubits:
            prob = results[qubit].probability()
            error = np.sqrt(prob * (1 - prob) / params.nshots)
            data.register_qubit(
                RamseyType,
                (qubit),
                dict(
                    wait=np.array([wait]),
                    prob=np.array([prob]),
                    errors=np.array([error]),
                ),
            )
    return data


ramsey_sequences = Routine(_acquisition, _fit, _plot, _update)
"""Ramsey Routine object."""
