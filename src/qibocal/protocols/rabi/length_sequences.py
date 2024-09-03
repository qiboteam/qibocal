import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Routine

from .length_signal import (
    RabiLengthSignalData,
    RabiLengthSignalParameters,
    RabiLenSignalType,
    _fit,
    _plot,
    _update,
)
from .utils import sequence_length


def _acquisition(
    params: RabiLengthSignalParameters, platform: Platform, targets: list[QubitId]
) -> RabiLengthSignalData:
    r"""
    Data acquisition for RabiLength Experiment.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.
    """

    sequence, qd_pulses, ro_pulses, amplitudes = sequence_length(
        targets, params, platform
    )

    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )

    data = RabiLengthSignalData(amplitudes=amplitudes)

    # sweep the parameter
    for duration in qd_pulse_duration_range:
        for qubit in targets:
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

        for qubit in targets:
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(
                RabiLenSignalType,
                (qubit),
                dict(
                    length=np.array([duration]),
                    signal=np.array([result.magnitude]),
                    phase=np.array([result.phase]),
                ),
            )

    return data


rabi_length_sequences = Routine(_acquisition, _fit, _plot, _update)
"""RabiLength Routine object."""
