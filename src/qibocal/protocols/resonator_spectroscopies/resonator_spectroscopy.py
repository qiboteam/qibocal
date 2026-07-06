from _collections_abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, fields

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    PowerLevel,
    Range,
    RangeLike,
    lorentzian_fit,
    lorentzian_with_linear_background,
    readout_frequency,
    to_range,
)
from qibocal.result import magnitude, phase
from qibocal.update import replace

from .resonator_utils import s21, s21_fit, s21_spectroscopy_plot, spectroscopy_plot

__all__ = ["resonator_spectroscopy", "ResonatorSpectroscopyData", "ResSpecType"]

ResSpecType = np.dtype(
    [
        ("freq", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator spectroscopy."""


@dataclass
class ResonatorSpectroscopyFit:
    """ResonatorSpectroscopy fit."""

    function: Callable
    """Protocol function to fit data with a model."""
    fit: Callable
    """Fit function used for the resonance."""
    values: Callable
    """Extract values from data."""
    plot: Callable
    """Plotting function for ResonatorSpectroscopy."""


FITS = {
    "lorentzian": ResonatorSpectroscopyFit(
        lorentzian_with_linear_background,
        lorentzian_fit,
        lambda z: z.signal,
        spectroscopy_plot,
    ),
    "s21": ResonatorSpectroscopyFit(
        s21,
        s21_fit,
        lambda z: (z.signal, z.phase),
        s21_spectroscopy_plot,
    ),
}
"""Dictionary of available fitting routines for ResonatorSpectroscopy."""


@dataclass
class ResonatorSpectroscopyParameters(Parameters):
    """ResonatorSpectroscopy runcard inputs."""

    frequency: RangeLike | None = None
    """Frequency range for sweep [Hz]."""
    freq_width: int | None = None
    """Width for frequency sweep relative  to the readout frequency [Hz].

    .. deprecated:: 0.2.6
        Use :attr:`frequency` instead.
    """
    freq_step: int | None = None
    """Frequency step for sweep [Hz].

    .. deprecated:: 0.2.6
        Use :attr:`frequency` instead.
    """
    power_level: PowerLevel | str = PowerLevel.low
    """Power regime (low or high). If low the readout frequency will be updated.
    If high both the readout frequency and the bare resonator frequency will be updated."""
    fit_function: str = "lorentzian"
    """Protocol function (lorentzian or s21) to fit data with a model."""
    phase_sign: bool = True
    """Several instruments have their convention about the sign of the phase. If True, the routine
    will apply a minus to the phase data."""
    amplitude: float | None = None
    """Readout amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""

    def __post_init__(self):
        if isinstance(self.power_level, str):
            self.power_level = PowerLevel(self.power_level)

    def frequency_range(self, q: QubitId, platform: CalibrationPlatform) -> Range:
        try:
            center = readout_frequency(q, platform, self.power_level)
        except KeyError:
            center = 0.0
        return (
            to_range(self.frequency, center=center)
            if self.frequency is not None
            else (
                center - self.freq_width / 2,
                center + self.freq_width / 2,
                self.freq_step,
            )
        )


@dataclass
class ResonatorSpectroscopyResults(Results):
    """ResonatorSpectroscopy outputs."""

    frequency: dict[QubitId, float]
    """Readout frequency [Hz] for each qubit."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitted parameters."""
    bare_frequency: dict[QubitId, float] | None = field(
        default_factory=dict,
    )
    """Bare resonator frequency [Hz] for each qubit."""
    amplitude: dict[QubitId, float] | None = field(
        default_factory=dict,
    )
    """Readout amplitude for each qubit."""

    def __contains__(self, key: QubitId):
        return all(
            key in getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), dict)
            and field.name != "bare_frequency"
        )


@dataclass
class ResonatorSpectroscopyData(Data):
    """Data structure for resonator spectroscopy with attenuation."""

    resonator_type: str
    """Resonator type."""
    amplitudes: dict[QubitId, float]
    """Amplitudes provided by the user."""
    fit_function: str = "lorentzian"
    """Fit function (optional) used for the resonance."""
    phase_sign: bool = False
    """Several instruments have their convention about the sign of the phase. If True, the routine
    will apply a minus to the phase data."""
    data: dict[QubitId, npt.NDArray[ResSpecType]] = field(default_factory=dict)
    """Raw data acquired."""
    power_level: PowerLevel = PowerLevel.low
    """Power regime of the resonator."""


def _acquisition(
    params: ResonatorSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorSpectroscopyData:
    """Data acquisition for resonator spectroscopy."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    amplitudes = {}

    for q in targets:
        natives = platform.natives.single_qubit[q]
        channel, pulse = natives.MZ()[0]

        if params.amplitude is not None:
            probe = replace(pulse.probe, amplitude=params.amplitude)
            pulse = replace(pulse, probe=probe)

        amplitudes[q] = pulse.probe.amplitude

        ro_pulses[q] = pulse
        sequence.append((channel, pulse))

    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            range=params.frequency_range(q, platform),
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]

    data = ResonatorSpectroscopyData(
        resonator_type=platform.resonator_type,
        power_level=params.power_level,
        amplitudes=amplitudes,
        fit_function=params.fit_function,
        phase_sign=params.phase_sign,
    )

    results = platform.execute(
        [sequence],
        [sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # retrieve the results for every qubit
    for q in targets:
        result = results[ro_pulses[q].id]
        # store the results
        signal = magnitude(result)
        phase_ = phase(result)
        data.register_qubit(
            ResSpecType,
            q,
            dict(
                signal=signal,
                phase=phase_,
                freq=np.arange(*params.frequency_range(q, platform)).tolist(),
            ),
        )
    return data


def _fit(
    data: ResonatorSpectroscopyData,
) -> ResonatorSpectroscopyResults:
    """Post-processing function for ResonatorSpectroscopy."""
    # TODO: change data.qubits
    qubits = data.qubits
    bare_frequency = {}
    frequency = {}
    fitted_parameters = {}

    fit = FITS[data.fit_function]

    for qubit in qubits:
        qubit_data = deepcopy(data[qubit])
        qubit_data.phase = (
            -qubit_data.phase if data.phase_sign else qubit_data.phase
        )  # TODO: tmp patch for the sign of the phase
        qubit_data.phase = np.unwrap(
            qubit_data.phase
        )  # TODO: move phase unwrapping in qibolab
        fit_result = fit.fit(
            qubit_data, resonator_type=data.resonator_type, fit="resonator"
        )
        if fit_result is not None:
            frequency[qubit], fitted_parameters[qubit], _ = fit_result
            if data.power_level == PowerLevel.high:
                bare_frequency[qubit] = frequency[qubit]

    return ResonatorSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        bare_frequency=bare_frequency if data.power_level == PowerLevel.high else {},
        amplitude=data.amplitudes,
    )


def _plot(
    data: ResonatorSpectroscopyData, target: QubitId, fit: ResonatorSpectroscopyResults
):
    """Plotting function for ResonatorSpectroscopy."""
    return FITS[data.fit_function].plot(data, target, fit)


def _update(
    results: ResonatorSpectroscopyResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.readout_frequency(results.frequency[target], platform, target)
    if len(results.bare_frequency) == 0:
        update.readout_amplitude(results.amplitude[target], platform, target)
        update.dressed_resonator_frequency(results.frequency[target], platform, target)

    else:
        update.bare_resonator_frequency(
            results.bare_frequency[target], platform, target
        )
        platform.calibration.single_qubits[
            target
        ].resonator.bare_frequency_amplitude = results.amplitude[target]


resonator_spectroscopy = Protocol(_acquisition, _fit, _plot, _update)
"""ResonatorSpectroscopy Protocol object."""
