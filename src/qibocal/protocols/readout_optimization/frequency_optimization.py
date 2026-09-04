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
    readout_frequency,
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

__all__ = ["ro_frequency"]


@dataclass
class ReadoutFrequencyParameters(Parameters):
    """Optimization RO frequency inputs."""

    frequency_range: RangeLike
    """Frequency RangeLike object; for further information, see
    :class:`qibocal.protocols.utils.RangeLike`."""
    save_iq: bool = False
    """Whether to save the IQ data during the acquisition."""


def _acquisition(
    params: ReadoutFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutData:
    r"""
    Data acquisition for readout frequency optimization.
    While sweeping the readout frequency, the routine performs a single shot
    classification and evaluates the assignment fidelity.
    At the end, the readout frequency is updated, choosing the one that has
    the highest assignment fidelity.
    """

    sequences, probe_pulses_dict = readout_sequence(platform, targets)

    sweepers: dict[QubitId, Sweeper] = {}
    frequency_values: dict[QubitId, list[float]] = {}
    for qubit in targets:
        sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            range=to_range(
                spec=params.frequency_range, center=readout_frequency(qubit, platform)
            ),
            channels=[platform.qubits[qubit].probe],
        )
        frequency_values[qubit] = sweepers[qubit].values.astype(float).tolist()

    results = platform.execute(
        sequences,
        [list(sweepers.values())],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    data = ReadoutData(
        swept_parameter=frequency_values,
        save_iq=params.save_iq,
    )
    data.data = save_data(
        targets=targets,
        parameter_dict=frequency_values,
        pulses_dict=probe_pulses_dict,
        results=results,
        save_iq=params.save_iq,
    )

    return data


def _plot(data: ReadoutData, fit: ReadoutResults, target: QubitId):
    """Plotting function for Optimization RO frequency"""
    return readout_plot(data, fit, target, "Frequency [Hz]")


def _update(results: ReadoutResults, platform: CalibrationPlatform, target: QubitId):
    update.readout_frequency(results.best_swept_param[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)


ro_frequency = Protocol(_acquisition, readout_fit, _plot, _update)
"""Optimization RO frequency Protocol object"""
