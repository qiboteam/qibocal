from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log

from ..utils import chi2_reduced
from . import utils


@dataclass
class RabiAmplitudeParameters(Parameters):
    """RabiAmplitude runcard inputs."""

    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    pulse_length: Optional[float]
    """RX pulse duration (ns)."""


@dataclass
class RabiAmplitudeResults(Results):
    """RabiAmplitude outputs."""

    amplitude: dict[QubitId, tuple[float, Optional[float]]]
    """Drive amplitude for each qubit."""
    length: dict[QubitId, tuple[float, Optional[float]]]
    """Drive pulse duration. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitted parameters."""
    chi2: dict[QubitId, tuple[float, Optional[float]]] = field(default_factory=dict)


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
    params: RabiAmplitudeParameters, platform: Platform, qubits: Qubits
) -> RabiAmplitudeData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    durations = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        if params.pulse_length is not None:
            qd_pulses[qubit].duration = params.pulse_length

        durations[qubit] = qd_pulses[qubit].duration
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

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
        [qd_pulses[qubit] for qubit in qubits],
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
    for qubit in qubits:
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
    chi2 = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        x = qubit_data.amp
        y = qubit_data.prob

        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        local_maxima = find_peaks(mags, threshold=10)[0]
        index = local_maxima[0] if len(local_maxima) > 0 else None
        # 0.5 hardcoded guess for less than one oscillation
        f = x[index] / (x[1] - x[0]) if index is not None else 0.5
        pguess = [0.5, 0.5, 1 / f, np.pi / 2]
        try:
            popt, perr = curve_fit(
                utils.rabi_amplitude_fit,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi],
                    [1, 1, np.inf, np.pi],
                ),
                sigma=qubit_data.error,
            )
            perr = np.sqrt(np.diag(perr))
            pi_pulse_parameter = np.abs(popt[2] / 2)

        except:
            log.warning("rabi_fit: the fitting was not succesful")
            pi_pulse_parameter = 0
            popt = [0] * 4
            perr = [1] * 4

        pi_pulse_amplitudes[qubit] = (pi_pulse_parameter, perr[2] / 2)
        fitted_parameters[qubit] = popt.tolist()
        durations = {key: (value, 0) for key, value in data.durations.items()}
        chi2[qubit] = (
            chi2_reduced(
                y,
                utils.rabi_amplitude_fit(x, *popt),
                qubit_data.error,
            ),
            np.sqrt(2 / len(y)),
        )
    return RabiAmplitudeResults(pi_pulse_amplitudes, durations, fitted_parameters, chi2)


def _plot(data: RabiAmplitudeData, qubit, fit: RabiAmplitudeResults = None):
    """Plotting function for RabiAmplitude."""
    return utils.plot_probabilities(data, qubit, fit)


def _update(results: RabiAmplitudeResults, platform: Platform, qubit: QubitId):
    update.drive_amplitude(results.amplitude[qubit], platform, qubit)


rabi_amplitude = Routine(_acquisition, _fit, _plot, _update)
"""RabiAmplitude Routine object."""
