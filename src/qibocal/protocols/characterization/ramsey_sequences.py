import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Qubits, Routine

from .ramsey import RamseyData, RamseyParameters, RamseyType, _fit, _plot, _update


def _acquisition(
    params: RamseyParameters,
    platform: Platform,
    qubits: Qubits,
) -> RamseyData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    ro_pulses = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
    freqs = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit,
            start=RX90_pulses1[qubit].finish,
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        freqs[qubit] = qubits[qubit].drive_frequency
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include wait time and t_max
    data = RamseyData(
        n_osc=params.n_osc,
        t_max=params.delay_between_pulses_end,
        detuning_sign=+1,
        qubit_freqs=freqs,
    )

    # sweep the parameter
    for wait in waits:
        for qubit in qubits:
            RX90_pulses2[qubit].start = RX90_pulses1[qubit].finish + wait
            ro_pulses[qubit].start = RX90_pulses2[qubit].finish
            if params.n_osc != 0:
                # FIXME: qblox will induce a positive detuning with minus sign
                RX90_pulses2[qubit].relative_phase = (
                    RX90_pulses2[qubit].start
                    * data.detuning_sign
                    * 2
                    * np.pi
                    * (params.n_osc)
                    / params.delay_between_pulses_end
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
