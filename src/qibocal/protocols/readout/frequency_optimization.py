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
    base_sequence,
    fit_readout_classification_models,
    readout_fit,
    readout_plot,
)

__all__ = ["ro_frequency"]


@dataclass
class ReadoutFrequencyParameters(Parameters):
    """Optimization RO frequency inputs."""

    frequency_range: RangeLike
    """Frequency RangeLike object.

    For further information, see
    :class:`qibocal.protocols.utils.RangeLike`."""
    save_iq: bool = False
    """Whether to save the IQ data during the acquisition."""


def _acquisition(
    params: ReadoutFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutData:
    """
    Data acquisition for readout frequency optimization.
    """

    sequences, probe_pulses_dict = base_sequence(platform, targets)

    sweepers: list[Sweeper] = []
    frequency_values: dict[QubitId, list[float]] = {}
    for qubit in targets:
        sweeper = Sweeper(
            parameter=Parameter.frequency,
            range=to_range(
                spec=params.frequency_range, center=readout_frequency(qubit, platform)
            ),
            channels=[platform.qubits[qubit].probe],
        )
        frequency_values[qubit] = sweeper.values.tolist()
        sweepers.append(sweeper)

    results = platform.execute(
        sequences,
        [sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    data = fit_readout_classification_models(
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
"""Readout resonator frequency optimization protocol.


The protocol sweeps the probe frequency of the resonator probe pulse over the
range specified by ``ReadoutFrequencyParameters.frequency_range`` and
acquires integrated readout signals for each value. The acquired data are
fitted to identify the frequency that best separates the readout states.
The fit also determines the optimal IQ rotation angle and discrimination
threshold.

When updated, the selected frequency, IQ angle, and threshold are written to
the platform calibration for every target qubit. Set ``save_iq=True`` to
retain the acquired IQ data.
"""
