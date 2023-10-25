import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Qubits, Routine

from .length_msr import (
    RabiLengthVoltData,
    RabiLengthVoltParameters,
    RabiLenVoltType,
    _fit,
    _plot,
    _update,
)


def _acquisition(
    params: RabiLengthVoltParameters, platform: Platform, qubits: Qubits
) -> RabiLengthVoltData:
    r"""
    Data acquisition for RabiLength Experiment.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    amplitudes = {}
    for qubit in qubits:
        # TODO: made duration optional for qd pulse?
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
        if params.pulse_amplitude is not None:
            qd_pulses[qubit].amplitude = params.pulse_amplitude
        amplitudes[qubit] = qd_pulses[qubit].amplitude

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )

    data = RabiLengthVoltData(amplitudes=amplitudes)

    # sweep the parameter
    for duration in qd_pulse_duration_range:
        for qubit in qubits:
            qd_pulses[qubit].duration = duration
            ro_pulses[qubit].start = qd_pulses[qubit].finish

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
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(
                RabiLenVoltType,
                (qubit),
                dict(
                    length=np.array([duration]),
                    msr=np.array([result.magnitude]),
                    phase=np.array([result.phase]),
                ),
            )

    return data


rabi_length_sequences = Routine(_acquisition, _fit, _plot, _update)
"""RabiLength Routine object."""
