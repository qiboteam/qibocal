import numpy as np
from qibolab import AcquisitionType, AveragingMode, Platform

from qibocal.auto.operation import QubitId, Routine
from qibocal.result import magnitude, phase

from . import t1_signal
from .utils import CoherenceType


def _acquisition(
    params: t1_signal.T1SignalParameters, platform: Platform, targets: list[QubitId]
) -> t1_signal.T1SignalData:
    """Data acquisition for T1 sequences experiment.

    In this experiment the different delays are executing using a for loop on software.
    """

    ro_wait_range = np.arange(
        params.delay_before_readout_start,
        params.delay_before_readout_end,
        params.delay_before_readout_step,
    )

    data = t1_signal.T1SignalData()

    for wait in ro_wait_range:
        sequence, ro_pulses, _ = t1_signal.t1_sequence(
            platform=platform, targets=targets, delay=wait
        )

        results = platform.execute(
            [sequence],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        for qubit in targets:
            result = results[ro_pulses[qubit].id]
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


t1_sequences = Routine(_acquisition, t1_signal._fit, t1_signal._plot, t1_signal._update)
"""T1 Routine object."""
