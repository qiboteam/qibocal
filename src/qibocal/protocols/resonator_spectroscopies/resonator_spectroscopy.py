from _collections_abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude, phase
from qibocal.update import replace

from ..utils import (
    PowerLevel,
    chi2_reduced,
    chi2_reduced_complex,
    lorentzian,
    lorentzian_fit,
    readout_frequency,
)
from .resonator_utils import s21, s21_fit, s21_spectroscopy_plot, spectroscopy_plot

__all__ = ["resonator_spectroscopy", "ResonatorSpectroscopyData", "ResSpecType"]

ResSpecType = np.dtype(
    [
        ("freq", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
        ("error_signal", np.float64),
        ("error_phase", np.float64),
    ]
)
"""Custom dtype for resonator spectroscopy."""


@dataclass
class ResonatorSpectroscopyFit:
    """ResonatorSpectroscopy fit."""

    function: Callable
    """Routine function to fit data with a model."""
    fit: Callable
    """Fit function used for the resonance."""
    chi2: Callable
    """Chi2 reduced."""
    values: Callable
    """Extract values from data."""
    errors: Callable
    """Extract errors from data."""
    plot: Callable
    """Plotting function for ResonatorSpectroscopy."""


FITS = {
    "lorentzian": ResonatorSpectroscopyFit(
        lorentzian,
        lorentzian_fit,
        chi2_reduced,
        lambda z: z.signal,
        lambda z: z.error_signal,
        spectroscopy_plot,
    ),
    "s21": ResonatorSpectroscopyFit(
        s21,
        s21_fit,
        chi2_reduced_complex,
        lambda z: (z.signal, z.phase),
        lambda z: (z.error_signal, z.error_phase),
        s21_spectroscopy_plot,
    ),
}
"""Dictionary of available fitting routines for ResonatorSpectroscopy."""


@dataclass
class ResonatorSpectroscopyParameters(Parameters):
    """ResonatorSpectroscopy runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    power_level: Union[PowerLevel, str]
    """Power regime (low or high). If low the readout frequency will be updated.
    If high both the readout frequency and the bare resonator frequency will be updated."""
    fit_function: str = "lorentzian"
    """Routine function (lorentzian or s21) to fit data with a model."""
    phase_sign: bool = True
    """Several instruments have their convention about the sign of the phase. If True, the routine
    will apply a minus to the phase data."""
    amplitude: Optional[float] = None
    """Readout amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    hardware_average: bool = True
    """By default hardware average will be performed."""

    def __post_init__(self):
        if isinstance(self.power_level, str):
            self.power_level = PowerLevel(self.power_level)


@dataclass
class ResonatorSpectroscopyResults(Results):
    """ResonatorSpectroscopy outputs."""

    frequency: dict[QubitId, float]
    """Readout frequency [Hz] for each qubit."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitted parameters."""
    bare_frequency: Optional[dict[QubitId, float]] = field(
        default_factory=dict,
    )
    """Bare resonator frequency [Hz] for each qubit."""
    error_fit_pars: dict[QubitId, list] = field(default_factory=dict)
    """Errors of the fit parameters."""
    chi2_reduced: dict[QubitId, tuple[float, Optional[float]]] = field(
        default_factory=dict
    )
    """Chi2 reduced."""
    amplitude: Optional[dict[QubitId, float]] = field(
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

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=readout_frequency(q, platform, params.power_level)
            + delta_frequency_range,
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
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    # retrieve the results for every qubit
    for q in targets:
        result = results[ro_pulses[q].id]
        # store the results
        ro_frequency = readout_frequency(q, platform, params.power_level)
        signal = magnitude(result)
        phase_ = phase(result)
        data.register_qubit(
            ResSpecType,
            (q),
            dict(
                signal=signal.mean(axis=0),
                phase=phase_.mean(axis=0),
                freq=delta_frequency_range + ro_frequency,
                error_signal=np.std(signal, axis=0, ddof=1) / np.sqrt(signal.shape[0]),
                error_phase=np.std(phase_, axis=0, ddof=1) / np.sqrt(phase_.shape[0]),
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
    error_fit_pars = {}
    chi2 = {}
    amplitudes = {}

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
            (
                frequency[qubit],
                fitted_parameters[qubit],
                error_fit_pars[qubit],
            ) = fit_result

            dof = len(data[qubit].freq) - len(fitted_parameters[qubit])

            if data.power_level is PowerLevel.high:
                bare_frequency[qubit] = frequency[qubit]

            chi2[qubit] = (
                fit.chi2(
                    fit.values(data[qubit]),
                    fit.function(data[qubit].freq, *fitted_parameters[qubit]),
                    fit.errors(data[qubit]),
                    dof,
                ),
                np.sqrt(2 / dof),
            )
            amplitudes[qubit] = fitted_parameters[qubit][0]
    if data.power_level is PowerLevel.high:
        return ResonatorSpectroscopyResults(
            frequency=frequency,
            fitted_parameters=fitted_parameters,
            bare_frequency=bare_frequency,
            error_fit_pars=error_fit_pars,
            chi2_reduced=chi2,
            amplitude=data.amplitudes,
        )
    return ResonatorSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        error_fit_pars=error_fit_pars,
        chi2_reduced=chi2,
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


resonator_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorSpectroscopy Routine object."""
