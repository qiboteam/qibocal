from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    ParallelSweepers,
    Parameter,
    Sweeper,
)

from qibocal.auto.operation import Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ..utils import chi2_reduced
from .acquisition import check_correct_drive_lines_setup, sequence_amplitude
from .parent_classes import (
    RabiAmplitudeParameters,
    RabiData,
    RabiResults,
)
from .processing import (
    fit_amplitude_function,
    plot_probabilities,
    rabi_amplitude_function,
    rabi_initial_guess,
    update_rabi_parameters,
)

__all__ = ["rabi_amplitude"]


RabiAmpClassType = np.dtype(
    [("amp", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeClassificationData(RabiData):
    """RabiAmplitude data acquisition."""

    data: dict[QubitId, npt.NDArray[RabiAmpClassType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiAmplitudeClassificationData:
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

    data = RabiAmplitudeClassificationData(
        drive_lines={t: d for t, d in zip(targets, drive_lines)},
        rx90=params.rx90,
        durations=durations,
    )

    # sweep the parameter
    results = platform.execute(
        [sequence],
        [ParallelSweepers([sweeper])],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        prob = results[ro_pulse.id]
        data.register_qubit(
            RabiAmpClassType,
            (qubit),
            dict(
                amp=sweeper.values,
                prob=prob,
                error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
            ),
        )
    return data


def _fit(data: RabiAmplitudeClassificationData) -> RabiResults:
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

        pguess = rabi_initial_guess(x, y, "amp", signal=False)
        try:
            popt, perr, pi_pulse_parameter = fit_amplitude_function(
                x,
                y,
                pguess,
                sigma=qubit_data.error,
                signal=False,
            )
            pi_pulse_amplitudes[qubit] = [pi_pulse_parameter, perr[2] / 2]
            fitted_parameters[qubit] = popt
            durations = {key: [value, 0] for key, value in data.durations.items()}
            chi2[qubit] = [
                chi2_reduced(
                    y,
                    rabi_amplitude_function(x, *popt),
                    qubit_data.error,
                ),
                np.sqrt(2 / len(y)),
            ]

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")
    return RabiResults(
        drive_lines=data.drive_lines,
        length=durations,
        amplitude=pi_pulse_amplitudes,
        fitted_parameters=fitted_parameters,
        rx90=data.rx90,
        chi2=chi2,
    )


def _plot(
    data: RabiAmplitudeClassificationData,
    target: QubitId,
    fit: RabiResults | None = None,
):
    """Plotting function for RabiAmplitude."""
    return plot_probabilities(data, target, fit, data.rx90)


rabi_amplitude = Protocol(_acquisition, _fit, _plot, update_rabi_parameters)
"""RabiAmplitude Protocol object."""
