from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Routine
from qibocal.config import log

from ..utils import chi2_reduced, fallback_period, guess_period
from . import utils
from .amplitude_signal import RabiAmplitudeSignalParameters, RabiAmplitudeSignalResults


@dataclass
class RabiAmplitudeParameters(RabiAmplitudeSignalParameters):
    """RabiAmplitude runcard inputs."""


@dataclass
class RabiAmplitudeResults(RabiAmplitudeSignalResults):
    """RabiAmplitude outputs."""

    chi2: dict[QubitId, list[float]] = field(default_factory=dict)


RabiAmpType = np.dtype(
    [("amp", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeData(Data):
    """RabiAmplitude data acquisition."""

    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiAmpType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiAmplitudeParameters, platform: Platform, targets: list[QubitId]
) -> RabiAmplitudeData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    sequence, qd_pulses, _, durations = utils.sequence_amplitude(
        targets, params, platform
    )
    # define the parameter to sweep and its range:
    # qubit drive pulse amplitude
    qd_pulse_amplitude_range = np.arange(
        params.min_amp_factor,
        params.max_amp_factor,
        params.step_amp_factor,
    )
    sweeper = Sweeper(
        Parameter.amplitude,
        qd_pulse_amplitude_range,
        [qd_pulses[qubit] for qubit in targets],
        type=SweeperType.FACTOR,
    )

    data = RabiAmplitudeData(durations=durations)

    # sweep the parameter
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweeper,
    )
    for qubit in targets:
        prob = results[qubit].probability(state=1)
        data.register_qubit(
            RabiAmpType,
            (qubit),
            dict(
                amp=qd_pulses[qubit].amplitude * qd_pulse_amplitude_range,
                prob=prob.tolist(),
                error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
            ),
        )
    return data


def _fit(data: RabiAmplitudeData) -> RabiAmplitudeResults:
    """Post-processing for RabiAmplitude."""
    qubits = data.qubits

    pi_pulse_amplitudes = {}
    fitted_parameters = {}
    durations = {}
    chi2 = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        x = qubit_data.amp
        y = qubit_data.prob

        period = fallback_period(guess_period(x, y))
        pguess = [0.5, 0.5, period, np.pi]
        try:
            popt, perr, pi_pulse_parameter = utils.fit_amplitude_function(
                x,
                y,
                pguess,
                sigma=qubit_data.error,
                signal=False,
            )
            pi_pulse_amplitudes[qubit] = [pi_pulse_parameter, perr[2] / 2]
            fitted_parameters[qubit] = popt.tolist()
            durations = {key: [value, 0] for key, value in data.durations.items()}
            chi2[qubit] = [
                chi2_reduced(
                    y,
                    utils.rabi_amplitude_function(x, *popt),
                    qubit_data.error,
                ),
                np.sqrt(2 / len(y)),
            ]

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")
    return RabiAmplitudeResults(pi_pulse_amplitudes, durations, fitted_parameters, chi2)


def _plot(data: RabiAmplitudeData, target: QubitId, fit: RabiAmplitudeResults = None):
    """Plotting function for RabiAmplitude."""
    return utils.plot_probabilities(data, target, fit)


def _update(results: RabiAmplitudeResults, platform: Platform, target: QubitId):
    update.drive_amplitude(results.amplitude[target], platform, target)
    update.drive_duration(results.length[target], platform, target)


rabi_amplitude = Routine(_acquisition, _fit, _plot, _update)
"""RabiAmplitude Routine object."""
