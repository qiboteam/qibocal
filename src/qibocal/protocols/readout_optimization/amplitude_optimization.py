from dataclasses import dataclass

from qibolab import (
    AcquisitionType,
    Parameter,
    Sweeper,
)

from qibocal import update
from qibocal.auto.operation import Parameters, Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    RangeLike,
    to_range,
)

from .utils import (
    ReadoutData,
    ReadoutResults,
    readout_fit,
    readout_plot,
    readout_sequence,
    save_data,
)

__all__ = ["ro_amplitude"]


@dataclass
class ReadoutAmplitudeParameters(Parameters):
    """ReadoutAmplitude runcard inputs."""

    amplitude_range: RangeLike
    """Amplitude RangeLike object; for further information, see
    :class:`qibocal.protocols.utils.RangeLike`."""
    save_iq: bool = False
    """Whether to save the IQ data during the acquisition."""


def _acquisition(
    params: ReadoutAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutData:
    r"""
    Data acquisition for resoantor amplitude optmization.
    This protocol sweeps the readout amplitude performing a classification routine
    and evaluating the error probability at each step. The sweep will be interrupted
    if the probability error is less than the `error_threshold`.
    """

    sequences, probe_pulses_dict = readout_sequence(platform, targets)

    sweepers: dict[QubitId, Sweeper] = {}
    amplitude_values: dict[QubitId, list[float]] = {}
    for qubit in targets:
        sweepers[qubit] = Sweeper(
            parameter=Parameter.amplitude,
            range=to_range(
                spec=params.amplitude_range,
                center=probe_pulses_dict[qubit][0].probe.amplitude,
            ),
            pulses=list(probe_pulses_dict[qubit].values()),
        )
        amplitude_values[qubit] = sweepers[qubit].values.astype(float).tolist()

    results = platform.execute(
        sequences,
        [list(sweepers.values())],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    data = ReadoutData(
        swept_parameter=amplitude_values,
        save_iq=params.save_iq,
    )
    data.data = save_data(
        targets=targets,
        parameter_dict=amplitude_values,
        pulses_dict=probe_pulses_dict,
        results=results,
        save_iq=params.save_iq,
    )

    return data


def _plot(data: ReadoutData, fit: ReadoutResults, target: QubitId):
    """Plotting function for Optimization RO amplitude."""
    return readout_plot(data, fit, target, "Amplitude [a.u.]")


def _update(results: ReadoutResults, platform: CalibrationPlatform, target: QubitId):
    update.readout_amplitude(results.best_swept_param[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


ro_amplitude = Protocol(_acquisition, readout_fit, _plot, _update)
"""Readout Amplitude Protocol  object."""
