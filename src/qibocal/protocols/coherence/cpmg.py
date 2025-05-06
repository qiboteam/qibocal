from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import probability

from .spin_echo import SpinEchoParameters, SpinEchoResults
from .t1 import CoherenceProbType, T1Data
from .utils import dynamical_decoupling_sequence, exponential_fit_probability, plot

__all__ = ["cpmg"]


@dataclass
class CpmgParameters(SpinEchoParameters):
    """Cpmg runcard inputs."""

    n: int = 1
    """Number of pi rotations."""


@dataclass
class CpmgResults(SpinEchoResults):
    """SpinEcho outputs."""


class CpmgData(T1Data):
    """SpinEcho acquisition outputs."""


def _acquisition(
    params: CpmgParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> CpmgData:
    """Data acquisition for Cpmg"""
    # create a sequence of pulses for the experiment:
    sequence, delays = dynamical_decoupling_sequence(
        platform, targets, n=params.n, kind="CPMG"
    )

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
        duration = platform.natives.single_qubit[q].RX()[0][1].duration
        durations.append(duration)
        assert (
            params.delay_between_pulses_start - params.n * duration
        ) / 2 / params.n >= 0, (
            f"Initial delay too short for qubit {q}, "
            f"minimum delay should be {params.n * duration}"
        )

    assert len(set(durations)) == 1, (
        "Cannot run on mulitple qubit with different RX duration."
    )

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=(wait_range - params.n * durations[0]) / 2 / params.n,
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

    data = CpmgData()
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


def _fit(data: CpmgData) -> CpmgResults:
    """Post-processing for Cpmg."""
    t2Echos, fitted_parameters, pcovs, chi2 = exponential_fit_probability(data)
    return CpmgResults(t2Echos, fitted_parameters, pcovs, chi2)


cpmg = Routine(_acquisition, _fit, plot)
"""Cpmg Routine object."""
