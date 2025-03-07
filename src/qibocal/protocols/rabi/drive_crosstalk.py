from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal import update
from qibocal.auto.operation import Data, QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import probability

from ..utils import chi2_reduced, fallback_period, guess_period
from . import utils
from .amplitude import (
    RabiAmplitudeData,
    RabiAmplitudeResults,
    RabiAmplitudeSignalParameters,
    RabiAmpType,
    _fit,
)


@dataclass
class DriveCrosstalkParameters(RabiAmplitudeSignalParameters):
    """RabiAmplitude runcard inputs."""

    target_qubit: Optional[QubitId] = None


@dataclass
class DriveCrosstalkResults(RabiAmplitudeResults):
    """RabiAmplitude outputs."""


@dataclass
class DriveCrosstalkData(RabiAmplitudeData):
    """RabiAmplitude data acquisition."""

    target_qubit: Optional[QubitId] = None


def _acquisition(
    params: DriveCrosstalkParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> DriveCrosstalkData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    sequence, qd_pulses, ro_pulses, durations = utils.sequence_amplitude(
        targets, params, platform, params.rx90, crosstalk=params.target_qubit
    )

    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.min_amp, params.max_amp, params.step_amp),
        pulses=[qd_pulses[qubit] for qubit in targets],
    )

    updates = []
    if params.target_qubit is not None:
        f = platform.config(platform.qubits[params.target_qubit].drive).frequency
        for qubit in targets:
            channel = platform.qubits[qubit].drive

            updates.append({channel: {"frequency": f}})

    data = DriveCrosstalkData(
        durations=durations, rx90=params.rx90, target_qubit=params.target_qubit
    )

    # sweep the parameter
    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
        updates=updates,
    )
    for qubit in targets:
        prob = probability(results[ro_pulses[qubit].id], state=1)
        data.register_qubit(
            RabiAmpType,
            (qubit),
            dict(
                amp=sweeper.values,
                prob=prob.tolist(),
                error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
            ),
        )
    return data


def _plot(data: DriveCrosstalkData, target: QubitId, fit: DriveCrosstalkResults = None):
    """Plotting function for RabiAmplitude."""
    fig, fitting = utils.plot_probabilities(data, target, fit, data.rx90)
    fig[0].update_layout(
        yaxis_title=f"Excited state probability of {data.target_qubit}",
    )
    return fig, fitting


def _update(
    results: DriveCrosstalkResults, platform: CalibrationPlatform, target: QubitId
):
    pass


drive_crosstalk = Routine(_acquisition, _fit, _plot, _update)
"""RabiAmplitude Routine object."""
