from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from qibocal import update
from qibocal.auto.operation import Parameters, Routine
from qibocal.config import log
from qibocal.protocols.rabi.length_signal import (
    RabiLengthVoltData,
    RabiLengthVoltResults,
)

from ..utils import chi2_reduced
from . import utils


@dataclass
class RabiLengthParameters(Parameters):
    """RabiLength runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    pulse_amplitude: Optional[float] = None
    """Pi pulse amplitude. Same for all qubits."""


@dataclass
class RabiLengthResults(RabiLengthVoltResults):
    """RabiLength outputs."""

    chi2: dict[QubitId, tuple[float, Optional[float]]] = field(default_factory=dict)


RabiLenType = np.dtype(
    [("length", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthData(RabiLengthVoltData):
    """RabiLength acquisition outputs."""

    data: dict[QubitId, npt.NDArray[RabiLenType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiLengthParameters, platform: Platform, targets: list[QubitId]
) -> RabiLengthData:
    r"""
    Data acquisition for RabiLength Experiment.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.
    """

    sequence, qd_pulses, _, amplitudes = utils.sequence_length(
        targets, params, platform
    )
    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )

    sweeper = Sweeper(
        Parameter.duration,
        qd_pulse_duration_range,
        [qd_pulses[qubit] for qubit in targets],
        type=SweeperType.ABSOLUTE,
    )

    data = RabiLengthData(amplitudes=amplitudes)

    # execute the sweep
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
            RabiLenType,
            (qubit),
            dict(
                length=qd_pulse_duration_range,
                prob=prob,
                error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
            ),
        )
    return data


def _fit(data: RabiLengthData) -> RabiLengthResults:
    """Post-processing for RabiLength experiment."""

    qubits = data.qubits
    fitted_parameters = {}
    durations = {}
    amplitudes = {}
    chi2 = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        raw_x = qubit_data.length
        min_x = np.min(raw_x)
        max_x = np.max(raw_x)
        y = qubit_data.prob

        x = (raw_x - min_x) / (max_x - min_x)
        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        local_maxima = find_peaks(mags, threshold=1)[0]
        index = local_maxima[0] if len(local_maxima) > 0 else None
        # 0.5 hardcoded guess for less than one oscillation
        f = x[index] / (x[1] - x[0]) if index is not None else 0.5
        pguess = [0.5, 0.5, 1 / f, 0, 0]

        try:
            popt, perr = curve_fit(
                utils.rabi_length_function,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi, 0],
                    [1, 1, np.inf, np.pi, np.inf],
                ),
                sigma=qubit_data.error,
            )

            translated_popt = [
                popt[0],
                popt[1] * np.exp(min_x * popt[4] / (max_x - min_x)),
                popt[2] * (max_x - min_x),
                popt[3] - 2 * np.pi * min_x / popt[2] / (max_x - min_x),
                popt[4] / (max_x - min_x),
            ]

            perr = np.sqrt(np.diag(perr))
            pi_pulse_parameter = (
                translated_popt[2]
                / 2
                * utils.period_correction_factor(phase=translated_popt[3])
            )
            durations[qubit] = (pi_pulse_parameter, perr[2] * (max_x - min_x) / 2)
            fitted_parameters[qubit] = translated_popt
            amplitudes = {key: (value, 0) for key, value in data.amplitudes.items()}
            chi2[qubit] = (
                chi2_reduced(
                    y,
                    utils.rabi_length_function(raw_x, *translated_popt),
                    qubit_data.error,
                ),
                np.sqrt(2 / len(y)),
            )
        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiLengthResults(durations, amplitudes, fitted_parameters, chi2)


def _update(results: RabiLengthResults, platform: Platform, target: QubitId):
    update.drive_duration(results.length[target], platform, target)


def _plot(data: RabiLengthData, fit: RabiLengthResults, target: QubitId):
    """Plotting function for RabiLength experiment."""
    return utils.plot_probabilities(data, target, fit)


rabi_length = Routine(_acquisition, _fit, _plot, _update)
"""RabiLength Routine object."""
