from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import probability

from .spin_echo_signal import (
    SpinEchoSignalParameters,
    SpinEchoSignalResults,
    update_spin_echo,
)
from .t1 import CoherenceProbType, T1Data
from .utils import dynamical_decoupling_sequence, exponential_fit_probability, plot

__all__ = ["SpinEchoParameters", "SpinEchoResults", "spin_echo"]


@dataclass
class SpinEchoParameters(SpinEchoSignalParameters):
    """SpinEcho runcard inputs."""


@dataclass
class SpinEchoResults(SpinEchoSignalResults):
    """SpinEcho outputs."""

    chi2: Optional[dict[QubitId, tuple[float, Optional[float]]]] = field(
        default_factory=dict
    )
    """Chi squared estimate mean value and error."""


class SpinEchoData(T1Data):
    """SpinEcho acquisition outputs."""


def _acquisition(
    params: SpinEchoParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> SpinEchoData:
    """Data acquisition for SpinEcho"""
    # create a sequence of pulses for the experiment:
    sequence, delays = dynamical_decoupling_sequence(platform, targets, kind="CP")

    # define the parameter to sweep and its range:
    # delay between pulses
    wait_range = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    durations = []
    for q in targets:
        # this is assuming that RX and RX90 have the same duration
        duration = platform.natives.single_qubit[q].RX().duration
        durations.append(duration)
        assert (params.delay_between_pulses_start - duration) / 2 >= 0, (
            f"Initial delay too short for qubit {q}, minimum delay should be {duration}"
        )
    assert len(set(durations)) == 1, (
        "Cannot run on mulitple qubit with different RX duration."
    )

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=(wait_range - durations[0]) / 2,
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

    data = SpinEchoData()
    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        result = results[ro_pulse.id]
        prob = probability(result, state=1)
        error = np.sqrt(prob * (1 - prob) / params.nshots)
        data.register_qubit(
            CoherenceProbType,
            (qubit),
            dict(
                wait=wait_range,
                prob=prob,
                error=error,
            ),
        )

    return data


def _fit(data: SpinEchoData) -> SpinEchoResults:
    """Post-processing for SpinEcho."""
    t2Echos, fitted_parameters, pcovs, chi2 = exponential_fit_probability(data)
    return SpinEchoResults(t2Echos, fitted_parameters, pcovs, chi2)


spin_echo = Routine(_acquisition, _fit, plot, update_spin_echo)
"""SpinEcho Routine object."""
