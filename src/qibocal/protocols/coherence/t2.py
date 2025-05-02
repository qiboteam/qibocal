from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform

from ...result import probability
from ..ramsey.utils import ramsey_sequence
from . import utils
from .t1 import CoherenceProbType, T1Data
from .t2_signal import T2SignalParameters, T2SignalResults, update_t2


@dataclass
class T2Parameters(T2SignalParameters):
    """T2 runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""


@dataclass
class T2Results(T2SignalResults):
    """T2 outputs."""

    chi2: Optional[dict[QubitId, tuple[float, Optional[float]]]] = field(
        default_factory=dict
    )
    """Chi squared estimate mean value and error."""


class T2Data(T1Data):
    """T2 acquisition outputs."""


def _acquisition(
    params: T2Parameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> T2Data:
    """Data acquisition for T2 experiment."""

    waits = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    sequence, delays = ramsey_sequence(platform, targets)

    data = T2Data()

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=waits,
        pulses=delays,
    )

    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        probs = probability(results[ro_pulse.id], state=1)
        errors = np.sqrt(probs * (1 - probs) / params.nshots)
        data.register_qubit(
            CoherenceProbType, (qubit), dict(wait=waits, prob=probs, error=errors)
        )
    return data


def _fit(data: T2Data) -> T2Results:
    """The used model is

    .. math::

        y = p_0 - p_1 e^{-x p_2}.
    """
    t2s, fitted_parameters, pcovs, chi2 = utils.exponential_fit_probability(data)
    return T2Results(t2s, fitted_parameters, pcovs, chi2)


t2 = Routine(_acquisition, _fit, utils.plot, update_t2)
"""T2 Routine object."""
