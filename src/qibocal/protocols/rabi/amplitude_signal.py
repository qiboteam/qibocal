from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ParallelSweepers, Parameter, Sweeper

from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import readout_frequency
from qibocal.result import collect, magnitude

from .acquisition import check_correct_drive_lines_setup, sequence_amplitude
from .parent_classes import (
    RabiAmplitudeParameters,
    RabiData,
    RabiResults,
)
from .processing import (
    fit_amplitude_function,
    plot_signal,
    rabi_initial_guess,
    update_rabi_ampl_params,
)

__all__ = ["rabi_amplitude_signal"]


RabiAmpSignalType = np.dtype(
    [("amp", np.float64), ("i", np.float64), ("q", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeSignalData(RabiData):
    """RabiAmplitudeSignal data acquisition."""

    data: dict[QubitId, npt.NDArray[RabiAmpSignalType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiAmplitudeSignalData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    drive_lines = check_correct_drive_lines_setup(
        targets=targets, input_drivelines=params.drive_lines
    )

    # create a sequence of pulses for the experiment
    sequence, qd_pulses, durations, updates = sequence_amplitude(
        targets=targets,
        drive_lines=drive_lines,
        platform=platform,
        pulse_duration=params.pulse_length,
        pulse_ampl=None,  # in this case we are sweeping on amplitude
        rx90=params.rx90,
    )

    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=params.amplitude_range,
        pulses=qd_pulses,
    )

    data = RabiAmplitudeSignalData(
        drive_lines={t: d for t, d in zip(targets, drive_lines)},
        rx90=params.rx90,
        durations=durations,
    )

    # for signal measurement we have to change readout
    updates |= {
        platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}
        for q in targets
    }

    # sweep the parameter
    results = platform.execute(
        [sequence],
        [ParallelSweepers([sweeper])],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        result = results[ro_pulse.id]
        data.register_qubit(
            RabiAmpSignalType,
            (qubit),
            dict(
                amp=sweeper.values,
                i=result[..., 0],
                q=result[..., 1],
            ),
        )
    return data


def _fit(data: RabiAmplitudeSignalData) -> RabiResults:
    """Post-processing for RabiAmplitude experiment."""

    qubits = data.qubits
    pi_pulse_amplitudes = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        rabi_parameter = qubit_data.amp
        voltages = magnitude(collect(i=qubit_data.i, q=qubit_data.q))

        y_min = np.min(voltages)
        y_max = np.max(voltages)
        x_min = np.min(rabi_parameter)
        x_max = np.max(rabi_parameter)
        x = (rabi_parameter - x_min) / (x_max - x_min)
        y = (voltages - y_min) / (y_max - y_min)

        pguess = rabi_initial_guess(x, y, "amp", signal=True)
        try:
            popt, _, pi_pulse_parameter = fit_amplitude_function(
                x,
                y,
                pguess,
                signal=True,
                x_limits=(x_min, x_max),
                y_limits=(y_min, y_max),
            )
            pi_pulse_amplitudes[qubit] = [pi_pulse_parameter]
            fitted_parameters[qubit] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiResults(
        drive_lines=data.drive_lines,
        length={k: [v] for k, v in data.durations.items()},
        amplitude=pi_pulse_amplitudes,
        fitted_parameters=fitted_parameters,
        rx90=data.rx90,
    )


def _plot(
    data: RabiAmplitudeSignalData,
    target: QubitId,
    fit: RabiResults | None = None,
):
    """Plotting function for RabiAmplitude."""
    return plot_signal(data, target, fit, data.rx90)


rabi_amplitude_signal = Protocol(_acquisition, _fit, _plot, update_rabi_ampl_params)
"""RabiAmplitude Protocol object."""
