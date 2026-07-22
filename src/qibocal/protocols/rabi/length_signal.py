from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ParallelSweepers, Parameter, Sweeper

from qibocal.auto.operation import Protocol, QubitId, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import readout_frequency
from qibocal.result import collect, magnitude

from .acquisition import define_qubits_and_drivelines, sequence_length
from .parent_classes import (
    RabiData,
    RabiLengthParameters,
    RabiResults,
)
from .processing import (
    fit_length_function,
    plot_signal,
    rabi_initial_guess,
    update_rabi_parameters,
)

__all__ = ["rabi_length_signal"]


RabiLenSignalType = np.dtype(
    [("length", np.float64), ("i", np.float64), ("q", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthSignalData(RabiData):
    """RabiLength acquisition outputs."""

    data: dict[QubitId, npt.NDArray[RabiLenSignalType]] = field(default_factory=dict)
    """Raw data acquired for classification experiment."""


def _acquisition(
    params: RabiLengthParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId] | list[QubitPairId],
) -> RabiLengthSignalData:
    r"""
    Data acquisition for RabiLength Signal Experiment.
    """

    qubits_list, drive_lines = define_qubits_and_drivelines(targets)

    sequence, qd_pulses, delays, amplitudes, updates = sequence_length(
        targets=qubits_list,
        drive_lines=drive_lines,
        platform=platform,
        pulse_ampl=params.pulse_amplitude,
        pulse_duration=None,  # in this case we are sweeping on duration
        rx90=params.rx90,
        use_align=params.interpolated_sweeper,
    )

    if params.interpolated_sweeper:
        # in this case delays is always an empty list, so it is safe to sum to qd_pulses
        sweep_param = Parameter.duration_interpolated
    else:
        sweep_param = Parameter.duration

    sweeper = Sweeper(
        parameter=sweep_param,
        range=params.duration_range,
        pulses=qd_pulses + delays,
    )

    data = RabiLengthSignalData(
        rx90=params.rx90,
        amplitudes=amplitudes,
    )

    # for signal measurement we have to change readout
    updates |= {
        platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}
        for q in qubits_list
    }

    results = platform.execute(
        [sequence],
        [ParallelSweepers([sweeper])],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for q in qubits_list:
        ro_pulse = list(sequence.channel(platform.qubits[q].acquisition))[-1]
        result = results[ro_pulse.id]
        data.register_qubit(
            RabiLenSignalType,
            (q),
            dict(
                length=sweeper.values,
                i=result[..., 0],
                q=result[..., 1],
            ),
        )
    return data


def _fit(data: RabiLengthSignalData) -> RabiResults:
    """Post-processing for RabiLength experiment."""

    qubits = data.qubits
    fitted_parameters = {}
    durations = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        rabi_parameter = qubit_data.length
        voltages = magnitude(collect(i=qubit_data.i, q=qubit_data.q))

        y_min = np.min(voltages)
        y_max = np.max(voltages)
        x_min = np.min(rabi_parameter)
        x_max = np.max(rabi_parameter)
        x = (rabi_parameter - x_min) / (x_max - x_min)
        y = (voltages - y_min) / (y_max - y_min) - 1 / 2

        pguess = rabi_initial_guess(x, y, "length", signal=True)

        try:
            popt, _, pi_pulse_parameter = fit_length_function(
                x,
                y,
                pguess,
                signal=True,
                x_limits=(x_min, x_max),
                y_limits=(y_min, y_max),
            )
            durations[qubit] = [pi_pulse_parameter]
            fitted_parameters[qubit] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiResults(
        length=durations,
        amplitude={k: [v] for k, v in data.amplitudes.items()},
        fitted_parameters=fitted_parameters,
        rx90=data.rx90,
    )


rabi_length_signal = Protocol(_acquisition, _fit, plot_signal, update_rabi_parameters)
"""RabiLength Protocol object.

In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
to find the drive pulse length that creates a rotation of a desired angle.
"""
