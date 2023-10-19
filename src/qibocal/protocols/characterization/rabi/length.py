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
class RabiLengthParameters(Parameters):
    """RabiLength runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration (ns)."""
    pulse_duration_end: float
    """Final pi pulse duration (ns)."""
    pulse_duration_step: float
    """Step pi pulse duration (ns)."""
    pulse_amplitude: Optional[float] = None
    """Pi pulse amplitude. Same for all qubits."""


@dataclass
class RabiLengthResults(Results):
    """RabiLength outputs."""

    length: dict[QubitId, tuple[int, Optional[float]]] = field(
        metadata=dict(update="drive_length")
    )
    """Pi pulse duration for each qubit."""
    amplitude: dict[QubitId, tuple[float, Optional[float]]] = field(
        metadata=dict(update="drive_amplitude")
    )
    """Pi pulse amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


RabiLenType = np.dtype(
    [("length", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthData(Data):
    """RabiLength acquisition outputs."""

    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiLenType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, length, prob, error):
        """Store output for single qubit."""
        # to be able to handle the non-sweeper case
        shape = (1,) if np.isscalar(length) else length.shape
        ar = np.empty(shape, dtype=RabiLenType)
        ar["length"] = length
        ar["prob"] = prob
        ar["error"] = error
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: RabiLengthParameters, platform: Platform, qubits: Qubits
) -> RabiLengthData:
    r"""
    Data acquisition for RabiLength Experiment.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    amplitudes = {}
    for qubit in qubits:
        # TODO: made duration optional for qd pulse?
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.pulse_duration_start
        )
        if params.pulse_amplitude is not None:
            qd_pulses[qubit].amplitude = params.pulse_amplitude
        amplitudes[qubit] = qd_pulses[qubit].amplitude

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

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
        [qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.ABSOLUTE,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse length
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

    for qubit in qubits:
        # average prob, phase, i and q over the number of shots defined in the runcard
        prob = results[qubit].probability(state=1)
        data.register_qubit(
            qubit,
            length=qd_pulse_duration_range,
            prob=prob,
            error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
        )
    return data


def _fit(data: RabiLengthData) -> RabiLengthResults:
    """Post-processing for RabiLength experiment."""

    qubits = data.qubits
    fitted_parameters = {}
    durations = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        x = qubit_data.length
        y = qubit_data.prob

        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        local_maxima = find_peaks(mags, threshold=1)[0]
        index = local_maxima[0] if len(local_maxima) > 0 else None
        # 0.5 hardcoded guess for less than one oscillation
        f = x[index] / (x[1] - x[0]) if index is not None else 0.5

        pguess = [1, 1, f, np.pi / 2, np.max(x) / 2]
        try:
            popt, perr = curve_fit(
                utils.rabi_length_fit,
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
            perr = np.sqrt(np.diag(perr))
            pi_pulse_parameter = np.abs(popt[2] / 2)
        except:
            log.warning("rabi_fit: the fitting was not succesful")
            pi_pulse_parameter = 0
            popt = [0] * 4 + [1]

        durations[qubit] = (pi_pulse_parameter, 0)
        fitted_parameters[qubit] = popt.tolist()
        amplitudes = {key: (value, 0) for key, value in data.amplitudes.items()}
    return RabiLengthResults(durations, amplitudes, fitted_parameters)


def _update(results: RabiLengthResults, platform: Platform, qubit: QubitId):
    update.drive_duration(results.length[qubit], platform, qubit)


def _plot(data: RabiLengthData, fit: RabiLengthResults, qubit):
    """Plotting function for RabiLength experiment."""
    return utils.plot_proba(data, qubit, fit)


rabi_length = Routine(_acquisition, _fit, _plot, _update)
"""RabiLength Routine object."""
