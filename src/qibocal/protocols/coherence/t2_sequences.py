import numpy as np
from qibolab import AcquisitionType, AveragingMode, Platform

from qibocal.auto.operation import QubitId, Routine

from ...result import magnitude, phase
from ..ramsey.utils import ramsey_sequence
from .t2_signal import T2SignalData, T2SignalParameters, _fit, _plot, _update
from .utils import CoherenceType


def _acquisition(
    params: T2SignalParameters,
    platform: Platform,
    targets: list[QubitId],
) -> T2SignalData:
    """Data acquisition for T2 experiment.

    In this experiment the different delays are executing using a for loop on software.

    """

    waits = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    data = T2SignalData()

    for wait in waits:
        sequence, _ = ramsey_sequence(platform, targets, wait=wait)
        results = platform.execute(
            [sequence],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
        for qubit in targets:
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            result = results[ro_pulse.id]
            data.register_qubit(
                CoherenceType,
                (qubit),
                dict(
                    wait=np.array([wait]),
                    signal=magnitude(np.array([result])),
                    phase=phase(np.array([result])),
                ),
            )
    return data


t2_sequences = Routine(_acquisition, _fit, _plot, _update)
"""T2 Routine object."""
