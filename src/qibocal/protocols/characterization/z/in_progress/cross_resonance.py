from dataclasses import dataclass
from typing import Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log

from .. import utils

# 2011 Gambetta - A simple all-microwave entangling gate for fixed-frequency superconducting qubits
# https://arxiv.org/pdf/1106.0553.pdf

# We need a RectangularGaussian pulse that it's duration can be swept in rt
# We need to be able to sweep multiple things symultaneously in rt (durations, pulse starts)


@dataclass
class CrossResonanceTomographyLengthParameters(Parameters):
    """Cross Resonance Tomography Length runcard inputs."""

    qubit_pairs: list[QubitPairId]
    """List of pairs of qubits (control_qubit, target_qubit)"""
    pulse_duration_start: float
    """Initial CR pulse duration [ns]."""
    pulse_duration_end: float
    """Final CR pulse duration [ns]."""
    pulse_duration_step: float
    """CR pulse duration sweep step [ns]."""
    pulse_amplitude: Optional[float] = None
    """CR pulse amplitude. Same for all qubits."""


@dataclass
class CrossResonanceTomographyLengthResults(Results):
    """Cross Resonance Tomography Length outputs."""


CrossResonanceTomographyLengthType = np.dtype(
    [("length", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for Cross Resonance Tomography Length."""


@dataclass
class CrossResonanceTomographyLengthData(Data):
    """Cross Resonance Tomography Length acquisition outputs."""


def _acquisition(
    params: CrossResonanceTomographyLengthParameters, platform: Platform, qubits: Qubits
) -> CrossResonanceTomographyLengthData:
    r"""
    Data acquisition for Cross Resonance Tomography Length Experiment.

    Cross Resonance Tomography Length experiment is the first of the experiments needed
    to characterise a CNOT Cross Resonance gate.
    In this experiment, the control qubit is prepared in either state |0> or in state |1>.
    After that, the CR pulse of varying duration is applied on the control qubit and
    the state of the target qubit is measured by projecting it onto the x, y and z bases.
    """

    # c q0)     P   - CR(wq1, t)    -
    # t q1)     I   - I             - T - M
    #
    # where:
    #   P is I or RX
    #   T is I, RX90, RY90

    # create a sequence of pulses for the experiment
    ps0x = PulseSequence()
    ps0y = PulseSequence()
    ps0z = PulseSequence()
    ps1x = PulseSequence()
    ps1y = PulseSequence()
    ps1z = PulseSequence()

    preparation_pulses = {}
    cr_pulses = {}
    tomography_x = {}
    tomography_y = {}
    tomography_z = {}
    ro_pulses = {}
    amplitudes = {}
    for control_qubit, target_qubit in params.qubit_pairs:
        preparation_pulses[control_qubit] = platform.create_RX_pulse(
            control_qubit, start=0
        )
        ps1x.add(preparation_pulses[control_qubit])
        ps1y.add(preparation_pulses[control_qubit])
        ps1z.add(preparation_pulses[control_qubit])

        # cr_ps = platform.create_CNOT_pulse_sequence([control_qubit, target_qubit])
        cr_pulse = platform.create_qubit_drive_pulse(
            qubit=target_qubit,
            start=preparation_pulses[control_qubit].finish,
            duration=params.pulse_duration_start,
        )
        cr_pulse.qubit = control_qubit

        if params.pulse_amplitude is not None:
            cr_pulse.amplitude = params.pulse_amplitude
        amplitudes[control_qubit] = cr_pulse.amplitude
        cr_pulses[control_qubit] = cr_pulse

        ps0x.add(cr_pulses[control_qubit])
        ps0y.add(cr_pulses[control_qubit])
        ps0z.add(cr_pulses[control_qubit])
        ps1x.add(cr_pulses[control_qubit])
        ps1y.add(cr_pulses[control_qubit])
        ps1z.add(cr_pulses[control_qubit])

        tomography_x[target_qubit] = platform.create_RX90_pulse(
            target_qubit,
            start=cr_pulses[control_qubit].finish,
            relative_phase=np.pi / 2,
        )

        tomography_y[target_qubit] = platform.create_RX90_pulse(
            target_qubit, start=cr_pulses[control_qubit].finish, relative_phase=-np.pi
        )

        tomography_z[target_qubit] = platform.create_RX90_pulse(
            target_qubit, start=cr_pulses[control_qubit].finish, relative_phase=0
        )
        tomography_z[target_qubit].amplitude = 0

        ps0x.add(tomography_x[target_qubit])
        ps1x.add(tomography_x[target_qubit])
        ps0y.add(tomography_y[target_qubit])
        ps1y.add(tomography_y[target_qubit])
        ps0z.add(tomography_z[target_qubit])
        ps1z.add(tomography_z[target_qubit])

        ro_pulses[target_qubit] = platform.create_qubit_readout_pulse(
            target_qubit, start=tomography_x[target_qubit].finish
        )

        ps0x.add(ro_pulses[target_qubit])
        ps0y.add(ro_pulses[target_qubit])
        ps0z.add(ro_pulses[target_qubit])
        ps1x.add(ro_pulses[target_qubit])
        ps1y.add(ro_pulses[target_qubit])
        ps1z.add(ro_pulses[target_qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )

    data = CrossResonanceTomographyLengthData(amplitudes=amplitudes)

    # execute the sweep
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
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            CrossResonanceTomographyLengthType,
            (qubit),
            dict(
                length=qd_pulse_duration_range,
                signal=result.magnitude,
                phase=result.phase,
            ),
        )
    return data


def _fit(
    data: CrossResonanceTomographyLengthData,
) -> CrossResonanceTomographyLengthResults:
    """Post-processing for Cross Resonance Tomography Length experiment."""

    qubits = data.qubits
    fitted_parameters = {}
    durations = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        rabi_parameter = qubit_data.length
        voltages = qubit_data.signal

        y_min = np.min(voltages)
        y_max = np.max(voltages)
        x_min = np.min(rabi_parameter)
        x_max = np.max(rabi_parameter)
        x = (rabi_parameter - x_min) / (x_max - x_min)
        y = (voltages - y_min) / (y_max - y_min) - 1 / 2

        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        local_maxima = find_peaks(mags, threshold=1)[0]
        index = local_maxima[0] if len(local_maxima) > 0 else None
        # 0.5 hardcoded guess for less than one oscillation
        f = x[index] / (x[1] - x[0]) if index is not None else 0.5

        pguess = [0, np.sign(y[0]) * 0.5, 1 / f, 0, 0]
        try:
            popt, _ = curve_fit(
                utils.rabi_length_function,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, -1, 0, -np.pi, 0],
                    [1, 1, np.inf, np.pi, np.inf],
                ),
            )
            translated_popt = [  # change it according to the fit function
                (y_max - y_min) * (popt[0] + 1 / 2) + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
                popt[2] * (x_max - x_min),
                popt[3] - 2 * np.pi * x_min / popt[2] / (x_max - x_min),
                popt[4] / (x_max - x_min),
            ]
            pi_pulse_parameter = (
                translated_popt[2]
                / 2
                * utils.period_correction_factor(phase=translated_popt[3])
            )

        except:
            log.warning("rabi_fit: the fitting was not succesful")
            pi_pulse_parameter = 0
            translated_popt = [0, 0, 1, 0, 0]
        durations[qubit] = pi_pulse_parameter
        fitted_parameters[qubit] = translated_popt

    return CrossResonanceTomographyLengthResults(
        durations, data.amplitudes, fitted_parameters
    )


def _update(
    results: CrossResonanceTomographyLengthResults, platform: Platform, qubit: QubitId
):
    update.drive_duration(results.length[qubit], platform, qubit)


def _plot(
    data: CrossResonanceTomographyLengthData,
    fit: CrossResonanceTomographyLengthResults,
    qubit,
):
    """Plotting function for Cross Resonance Tomography Length experiment."""
    return utils.plot(data, qubit, fit)


cross_resonance_tomography_length = Routine(_acquisition, _fit, _plot, _update)
"""Cross Resonance Tomography Length Routine object."""
