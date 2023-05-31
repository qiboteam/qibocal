from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine

from .utils import PowerLevel, lorentzian_fit, spectroscopy_plot


@dataclass
class ResonatorSpectroscopyParameters(Parameters):
    """ResonatorSpectroscopy runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    power_level: PowerLevel
    """Power regime (low or high). If low the readout frequency will be updated.
    If high both the readout frequency and the bare resonator frequency will be updated."""
    amplitude: Optional[float] = None
    """Readout amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    qubits: Optional[list] = field(default_factory=list)
    """Local qubits (optional)."""
    update: Optional[bool] = None
    """Runcard update mechanism."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""

    def __post_init__(self):
        # TODO: ask Alessandro if there is a proper way to pass Enum to class
        self.power_level = PowerLevel(self.power_level)


@dataclass
class ResonatorSpectroscopyResults(Results):
    """ResonatorSpectroscopy outputs."""

    frequency: Dict[QubitId, float] = field(metadata=dict(update="readout_frequency"))
    """Readout frequency [GHz] for each qubit."""
    fitted_parameters: Dict[QubitId, Dict[str, float]]
    """Raw fitted parameters."""
    bare_frequency: Optional[Dict[QubitId, float]] = field(
        default_factory=dict, metadata=dict(update="bare_resonator_frequency")
    )
    """Bare resonator frequency [GHz] for each qubit."""
    amplitude: Optional[Dict[QubitId, float]] = field(
        default_factory=dict, metadata=dict(update="readout_amplitude")
    )
    """Readout amplitude for each qubit."""


@dataclass
class GlobalParameters:
    qubits: list
    power_level: PowerLevel
    resonator_type: str


@dataclass
class QubitData:
    name: QubitId
    amplitude: float
    frequency: npt.NDArray[np.float64]
    voltage: npt.NDArray[np.float64]
    phase: npt.NDArray[np.float64]


@dataclass
class ResonatorSpectroscopyData:
    config: GlobalParameters
    measurement: Dict[QubitId, QubitData] = field(default_factory=dict)

    @classmethod
    def load_config(cls, **parameters):
        return cls(config=GlobalParameters(**parameters))

    def load_qubit(self, qubit, amplitude, voltage, phase, frequency):
        self.measurement[qubit] = QubitData(qubit, amplitude, frequency, voltage, phase)

    def to_csv(self, path):
        return


def _acquisition(
    params: ResonatorSpectroscopyParameters, platform: Platform, qubits: Qubits
) -> ResonatorSpectroscopyData:
    """Data acquisition for resonator spectroscopy."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    amplitudes = {}

    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        if params.amplitude is not None:
            ro_pulses[qubit].amplitude = params.amplitude

        amplitudes[qubit] = ro_pulses[qubit].amplitude

        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
    )
    data = ResonatorSpectroscopyData.load_config(
        qubits=[q for q in qubits],
        power_level=params.power_level,
        resonator_type=platform.resonator_type,
    )

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

    # retrieve the results for every qubit
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        # store the results
        data.load_qubit(
            qubit,
            amplitudes[qubit],
            result.magnitude,
            result.phase,
            delta_frequency_range + ro_pulses[qubit].frequency,
        )

    return data


def _fit(data: ResonatorSpectroscopyData) -> ResonatorSpectroscopyResults:
    """Post-processing function for ResonatorSpectroscopy."""
    qubits = data.config.qubits
    bare_frequency = {}
    amplitudes = {}
    frequency = {}
    fitted_parameters = {}
    for qubit in qubits:
        freq, fitted_params = lorentzian_fit(data, qubit)
        if data.config.power_level is PowerLevel.high:
            bare_frequency[qubit] = freq

        frequency[qubit] = freq
        amplitudes[qubit] = data.measurement[qubit].amplitude
        fitted_parameters[qubit] = fitted_params
    if data.config.power_level is PowerLevel.high:
        return ResonatorSpectroscopyResults(
            frequency=frequency,
            fitted_parameters=fitted_parameters,
            bare_frequency=bare_frequency,
            amplitude=amplitudes,
        )
    else:
        return ResonatorSpectroscopyResults(
            frequency=frequency,
            fitted_parameters=fitted_parameters,
            amplitude=amplitudes,
        )


def _plot(data: ResonatorSpectroscopyData, fit: ResonatorSpectroscopyResults, qubit):
    """Plotting function for ResonatorSpectroscopy."""
    return spectroscopy_plot(data, fit, qubit)


resonator_spectroscopy = Routine(_acquisition, _fit, _plot)
"""ResonatorSpectroscopy Routine object."""
