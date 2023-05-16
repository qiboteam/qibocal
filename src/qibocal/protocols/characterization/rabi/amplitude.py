from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits

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
    pulse_length: float
    """RX pulse duration (ns)."""
    nshots: int
    """Number of shots."""
    relaxation_time: float
    """Relaxation time (ns)."""


@dataclass
class RabiAmplitudeResults(Results):
    """RabiAmplitude outputs."""

    amplitude: Dict[List[Tuple], str] = field(metadata=dict(update="drive_amplitude"))
    """Drive amplitude for each qubit."""
    length: Dict[List[Tuple], str] = field(metadata=dict(update="drive_length"))
    """Drive pulse duration. Same for all qubits."""
    fitted_parameters: Dict[List[Tuple], List]
    """Raw fitted parameters."""


class RabiAmplitudeData(DataUnits):
    """RabiAmplitude data acquisition."""

    def __init__(self):
        super().__init__(
            "data",
            {"amplitude": "dimensionless", "length": "ns"},
            options=["qubit"],
        )


def _acquisition(
    params: RabiAmplitudeParameters, platform: AbstractPlatform, qubits: Qubits
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
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        qd_pulses[qubit].duration = params.pulse_length
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
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse amplitude
    data = RabiAmplitudeData()

    # sweep the parameter
    results = platform.sweep(
        sequence,
        sweeper,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        r = result.raw
        r.update(
            {
                "amplitude[dimensionless]": qd_pulses[qubit].amplitude
                * qd_pulse_amplitude_range,
                "length[ns]": len(qd_pulse_amplitude_range)
                * [qd_pulses[qubit].duration],
                "qubit": len(qd_pulse_amplitude_range) * [qubit],
            }
        )
        data.add_data_from_dict(r)
    return data


def _fit(data: RabiAmplitudeData) -> RabiAmplitudeResults:
    """Post-processing for RabiAmplitude."""
    qubits = data.df["qubit"].unique()

    pi_pulse_amplitudes = {}
    fitted_parameters = {}
    durations = {}

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == qubit]

        rabi_parameter = qubit_data["amplitude"].pint.to("dimensionless").pint.magnitude
        voltages = qubit_data["MSR"].pint.to("uV").pint.magnitude
        durations[qubit] = qubit_data["length"].pint.to("ns").pint.magnitude.unique()

        y_min = np.min(voltages.values)
        y_max = np.max(voltages.values)
        x_min = np.min(rabi_parameter.values)
        x_max = np.max(rabi_parameter.values)
        x = (rabi_parameter.values - x_min) / (x_max - x_min)
        y = (voltages.values - y_min) / (y_max - y_min)

        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        index = np.argmax(mags) if np.argmax(mags) != 0 else np.argmax(mags[1:]) + 1
        f = x[index] / (x[1] - x[0])

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

    return RabiAmplitudeResults(pi_pulse_amplitudes, durations, fitted_parameters)


def _plot(data: RabiAmplitudeData, fit: RabiAmplitudeResults, qubit):
    """Plotting function for RabiAmplitude."""
    return utils.plot(data, fit, qubit)


rabi_amplitude = Routine(_acquisition, _fit, _plot)
"""RabiAmplitude Routine object."""
