import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Qubits, Routine

from . import t1, t1_msr


def _acquisition(
    params: t1_msr.T1MSRParameters, platform: Platform, qubits: Qubits
) -> t1_msr.T1MSRData:
    r"""Data acquisition for T1 experiment.
    In a T1 experiment, we measure an excited qubit after a delay. Due to decoherence processes
    (e.g. amplitude damping channel), it is possible that, at the time of measurement, after the delay,
    the qubit will not be excited anymore. The larger the delay time is, the more likely is the qubit to
    fall to the ground state. The goal of the experiment is to characterize the decay rate of the qubit
    towards the ground state.

    Args:
        params:
        platform (Platform): Qibolab platform object
        qubits (list): list of target qubits to perform the action
        delay_before_readout_start (int): Initial time delay before ReadOut
        delay_before_readout_end (list): Maximum time delay before ReadOut
        delay_before_readout_step (int): Scan range step for the delay before ReadOut
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points
    """

    # create a sequence of pulses for the experiment
    # RX - wait t - MZ
    qd_pulses = {}
    ro_pulses = {}
    sequence = PulseSequence()
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # wait time before readout
    ro_wait_range = np.arange(
        params.delay_before_readout_start,
        params.delay_before_readout_end,
        params.delay_before_readout_step,
    )

    # create a DataUnits object to store the MSR, phase, i, q and the delay time
    data = t1_msr.T1MSRData()

    # repeat the experiment as many times as defined by software_averages
    # sweep the parameter
    for wait in ro_wait_range:
        for qubit in qubits:
            ro_pulses[qubit].start = qd_pulses[qubit].duration + wait

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )
        for qubit in qubits:
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(
                t1_msr.CoherenceType,
                (qubit),
                dict(
                    wait=np.array([wait]),
                    msr=np.array([result.magnitude]),
                    phase=np.array([result.phase]),
                ),
            )
    return data


t1_sequences = Routine(_acquisition, t1_msr._fit, t1_msr._plot, t1._update)
"""T1 Routine object."""
