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
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class RabiAmplitudeResults(Results):
    """RabiAmplitude outputs."""

    amplitude: dict[QubitId, float] = field(metadata=dict(update="drive_amplitude"))
    """Drive amplitude for each qubit."""
    length: dict[QubitId, float] = field(metadata=dict(update="drive_length"))
    """Drive pulse duration. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitted parameters."""


RabiAmpType = np.dtype(
    [("amp", np.float64), ("msr", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeData(Data):
    """RabiAmplitude data acquisition."""

    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiAmpType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, amp, msr, phase):
        """Store output for single qubit."""
        ar = np.empty(amp.shape, dtype=RabiAmpType)
        ar["amp"] = amp
        ar["msr"] = msr
        ar["phase"] = phase
        self.data[qubit] = np.rec.array(ar)


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

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse amplitude
    data = RabiAmplitudeData(durations=durations)

    # sweep the parameter
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit,
            amp=qd_pulses[qubit].amplitude * qd_pulse_amplitude_range,
            msr=result.magnitude,
            phase=result.phase,
        )
    return data


def _fit(data: RabiAmplitudeData) -> RabiAmplitudeResults:
    """Post-processing for RabiAmplitude."""
    qubits = data.qubits

    pi_pulse_amplitudes = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        rabi_parameter = qubit_data.amp
        voltages = qubit_data.msr

        y_min = np.min(voltages)
        y_max = np.max(voltages)
        x_min = np.min(rabi_parameter)
        x_max = np.max(rabi_parameter)
        x = (rabi_parameter - x_min) / (x_max - x_min)
        y = (voltages - y_min) / (y_max - y_min)

        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        local_maxima = find_peaks(mags, threshold=10)[0]
        index = local_maxima[0] if len(local_maxima) > 0 else None
        # 0.5 hardcoded guess for less than one oscillation
        f = x[index] / (x[1] - x[0]) if index is not None else 0.5
        pguess = [0.5, 1, f, np.pi / 2]
        try:
            popt, _ = curve_fit(
                utils.rabi_amplitude_fit,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi],
                    [1, 1, np.inf, np.pi],
                ),
            )
            translated_popt = [
                y_min + (y_max - y_min) * popt[0],
                (y_max - y_min) * popt[1],
                popt[2] / (x_max - x_min),
                popt[3] - 2 * np.pi * x_min / (x_max - x_min) * popt[2],
            ]
            pi_pulse_parameter = np.abs((1.0 / translated_popt[2]) / 2)

        except:
            log.warning("rabi_fit: the fitting was not succesful")
            pi_pulse_parameter = 0
            fitted_parameters = [0] * 4

        pi_pulse_amplitudes[qubit] = pi_pulse_parameter
        fitted_parameters[qubit] = translated_popt

    return RabiAmplitudeResults(pi_pulse_amplitudes, data.durations, fitted_parameters)


def _plot(data: RabiAmplitudeData, qubit, fit: RabiAmplitudeResults = None):
    """Plotting function for RabiAmplitude."""
    return utils.plot(data, qubit, fit)


def _update(results: RabiAmplitudeResults, platform: Platform, qubit: QubitId):
    update.drive_amplitude(results.amplitude[qubit], platform, qubit)


rabi_amplitude = Routine(_acquisition, _fit, _plot, _update)
"""RabiAmplitude Routine object."""
