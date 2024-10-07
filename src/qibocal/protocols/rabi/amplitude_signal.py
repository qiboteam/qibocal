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
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log

from . import utils

PROJECTIONS = ['Z', 'Y', 'X']
"""Standard projections for measurements."""


@dataclass
class RabiAmplitudeVoltParameters(Parameters):
    """RabiAmplitude runcard inputs."""

    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    pulse_length: Optional[float]
    """RX pulse duration [ns]."""
    projections: Optional[list[str]] = field(default_factory=lambda: [PROJECTIONS[0]])


@dataclass
class RabiAmplitudeVoltResults(Results):
    """RabiAmplitude outputs."""

    amplitude: dict[QubitId, tuple[float, Optional[float]]]
    """Drive amplitude for each qubit."""
    length: dict[QubitId, tuple[float, Optional[float]]]
    """Drive pulse duration. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitted parameters."""


RabiAmpVoltType = np.dtype(
    [("amp", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeVoltData(Data):
    """RabiAmplitudeVolt data acquisition."""

    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiAmpVoltType]] = field(default_factory=dict)
    """Raw data acquired."""


def ro_projection_pulse(platform: Platform, qubit, start=0, projection = PROJECTIONS[0]):
    """Create a readout pulse for a given qubit."""
    qd_pulse = platform.create_RX90_pulse(qubit, start=start)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=qd_pulse.finish)

    if projection == PROJECTIONS[0]:   
        qd_pulse.amplitude = 0
    elif projection == PROJECTIONS[1]:
        qd_pulse.relative_phase=0
    elif projection == PROJECTIONS[2]:
        qd_pulse.relative_phase=180
    else:
        raise ValueError(f"Invalid measurement <{projection}>")
    
    
    return qd_pulse, ro_pulse


def _acquisition(
    params: RabiAmplitudeVoltParameters, platform: Platform, targets: list[QubitId]
) -> RabiAmplitudeVoltData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    
    qd_pulses = {}
    ro_pulses = {}
    durations = {}
    projection_pulse = {}

    data = RabiAmplitudeVoltData()

    for projection in params.projections:
        sequence = PulseSequence()
        for qubit in targets:
            qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
            if params.pulse_length is not None:
                qd_pulses[qubit].duration = params.pulse_length

            durations[qubit] = qd_pulses[qubit].duration
            projection_pulse[qubit], ro_pulses[qubit] = ro_projection_pulse(
                platform, qubit, start=qd_pulses[qubit].finish, projection=projection  
            )
            dummy_qd = platform.create_RX90_pulse(qubit, start=0)
            dummy_qd.amplitude = 0
            dummy_qd.duration = 4
            #sequence.add(dummy_qd)
            
            sequence.add(qd_pulses[qubit])
            sequence.add(projection_pulse[qubit])
            sequence.add(ro_pulses[qubit])

            print(sequence)

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

        data.durations=durations

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

        for qubit in targets:
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(
                RabiAmpVoltType,
                (qubit, projection),
                dict(
                    amp=qd_pulses[qubit].amplitude * qd_pulse_amplitude_range,
                    signal=result.magnitude,
                    phase=result.phase,
                ),
            )
    
    return data


def _fit(data: RabiAmplitudeVoltData) -> RabiAmplitudeVoltResults:
    """Post-processing for RabiAmplitude."""
    qubits = data.qubits

    pi_pulse_amplitudes = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[(qubit, PROJECTIONS[0])]

        rabi_parameter = qubit_data.amp
        voltages = qubit_data.signal

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
        pguess = [0.5, 1, 1 / f, 0]
        try:
            popt, _ = curve_fit(
                utils.rabi_amplitude_function,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi],
                    [1, 1, np.inf, np.pi],
                ),
            )
            translated_popt = [  # Change it according to fit function changes
                y_min + (y_max - y_min) * popt[0],
                (y_max - y_min) * popt[1],
                popt[2] * (x_max - x_min),
                popt[3] - 2 * np.pi * x_min / (x_max - x_min) / popt[2],
            ]
            pi_pulse_parameter = (
                translated_popt[2]
                / 2
                * utils.period_correction_factor(phase=translated_popt[3])
            )
            pi_pulse_amplitudes[qubit] = pi_pulse_parameter
            fitted_parameters[qubit] = translated_popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiAmplitudeVoltResults(
        pi_pulse_amplitudes, data.durations, fitted_parameters
    )


def _plot(
    data: RabiAmplitudeVoltData, target: QubitId, fit: RabiAmplitudeVoltResults = None
):
    """Plotting function for RabiAmplitude."""

    figs = []
    fit_report = None
    for projection in PROJECTIONS:
        if ((target, projection)) not in data.data:
            continue
        if projection==PROJECTIONS[0]:
            fig, fit_report = utils.plot(data, target, fit , projection=projection)
        else:
            fig, _ = utils.plot(data, target, fit = None, projection=projection)
        figs.extend(fig)

    return figs, fit_report

def _update(results: RabiAmplitudeVoltResults, platform: Platform, target: QubitId):
    update.drive_amplitude(results.amplitude[target], platform, target)


rabi_amplitude_signal = Routine(_acquisition, _fit, _plot, _update)
"""RabiAmplitude Routine object."""
