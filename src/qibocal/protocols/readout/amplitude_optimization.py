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
    base_sequence,
    fit_readout_classification_models,
    readout_fit,
    readout_plot,
)

__all__ = ["ro_amplitude"]


@dataclass
class ReadoutAmplitudeParameters(Parameters):
    """ReadoutAmplitude runcard inputs."""

    amplitude_range: RangeLike
    """Amplitude RangeLike object.

    For further information, see
    :class:`qibocal.protocols.utils.RangeLike`."""
    save_iq: bool = False
    """Whether to save the IQ data during the acquisition."""


def _acquisition(
    params: ReadoutAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ReadoutData:
    """
    Data acquisition for resonator amplitude optimization.
    """

    sequences, probe_pulses_dict = base_sequence(platform, targets)

    sweepers: list[Sweeper] = []
    amplitude_values: dict[QubitId, list[float]] = {}
    for qubit in targets:
        _, ro_pulse = platform.parameters.native_gates.single_qubit[qubit].MZ()[0]
        sweeper = Sweeper(
            parameter=Parameter.amplitude,
            range=to_range(
                spec=params.amplitude_range,
                center=ro_pulse.probe.amplitude,
            ),
            pulses=list(probe_pulses_dict[qubit].values()),
        )
        amplitude_values[qubit] = sweeper.values.tolist()
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
"""Optimize the readout pulse amplitude for each target qubit.

The protocol sweeps the amplitude of the resonator probe pulse over the
range specified by ``ReadoutAmplitudeParameters.amplitude_range`` and
acquires integrated readout signals for each value. The acquired data are
fitted to identify the amplitude that best separates the readout states.
The fit also determines the optimal IQ rotation angle and discrimination
threshold.

When updated, the selected amplitude, IQ angle, and threshold are written to
the platform calibration for every target qubit. Set ``save_iq=True`` to
retain the acquired IQ data.
"""
