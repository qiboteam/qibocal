from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.protocols.characterization.qubit_spectroscopy_ef import (
    DEFAULT_ANHARMONICITY,
)


@dataclass
class SpuriousParameters(Parameters):
    """Spurious Identification runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the qubit frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [V]."""
    bias_step: float
    """Bias step for sweep (V)."""
    drive_amplitude: Optional[float] = None
    """Drive amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    transition: Optional[str] = "01"
    """Flux spectroscopy transition type ("01" or "02"). Default value is 01"""
    drive_duration: int = 2000
    """Duration of the drive pulse."""
    if_bandwidth: int = 500_000_000
    """Control device IF bandwidth."""
    lo_freq_width: int = 50
    """Width for LO frequency sweep relative to the qubit transition frequency (Hz).
    Its value should be less than the control device IF bandwidth"""
    lo_freq_step: int = 50
    """Frequency step for LO frequency sweep (Hz)."""


SpuriousType = np.dtype(
    [
        ("lo_freq", np.float64),
        ("freq", np.float64),
        ("bias", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator Spurious Identification."""


@dataclass
class SpuriousData(Data):
    """Spurious acquisition outputs."""

    data: dict[QubitId, npt.NDArray[SpuriousType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(
        self, qubit, lo_freq, frequencies, biases, msr_matrix, phase_matrix
    ):
        """Store output for single qubit."""

        size = len(frequencies) * len(biases)
        ar = np.empty(size, dtype=SpuriousType)
        frequency_matrix, bias_matrix = np.meshgrid(frequencies, biases)
        ar["lo_freq"] = np.full(shape=size, fill_value=lo_freq)
        ar["freq"] = frequency_matrix.ravel()
        ar["bias"] = bias_matrix.ravel()
        ar["msr"] = msr_matrix.ravel()
        ar["phase"] = phase_matrix.ravel()

        if not qubit in self.data:
            self.data[qubit] = np.rec.array(ar)
        else:
            self.data[qubit] = np.rec.array(np.append(self.data[qubit], ar))


@dataclass
class SpuriousResults(Results):
    """Spurious Identification outputs."""

    pass


def _acquisition(
    params: SpuriousParameters,
    platform: Platform,
    qubits: Qubits,
) -> SpuriousData:
    """Data acquisition for Spurious Identification."""
    data = SpuriousData()
    for qubit in qubits:  # one qubit at a time
        sequence = PulseSequence()
        ro_pulses = {}
        qd_pulses = {}

        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )

        if params.transition == "02":
            if qubits[qubit].anharmonicity:
                center_freq = (
                    qd_pulses[qubit].frequency - qubits[qubit].anharmonicity / 2
                )
            else:
                center_freq = qd_pulses[qubit].frequency - DEFAULT_ANHARMONICITY / 2
            qd_pulses[qubit].frequency = center_freq
        else:
            center_freq = qd_pulses[qubit].frequency

        lo_freq_range = center_freq + np.arange(
            -params.lo_freq_width // 2, params.lo_freq_width // 2, params.lo_freq_step
        )

        if params.drive_amplitude is not None:
            qd_pulses[qubit].amplitude = params.drive_amplitude

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

        # define the parameters to sweep and their range:
        delta_frequency_range = np.arange(
            -params.freq_width // 2, params.freq_width // 2, params.freq_step
        )
        freq_sweeper = Sweeper(
            Parameter.frequency,
            delta_frequency_range,
            pulses=[qd_pulses[qubit]],
            type=SweeperType.OFFSET,
        )

        delta_bias_range = np.arange(
            -params.bias_width / 2, params.bias_width / 2, params.bias_step
        )
        bias_sweeper = Sweeper(
            Parameter.bias,
            delta_bias_range,
            qubits=[qubits[qubit]],
            type=SweeperType.OFFSET,
        )

        options = ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        for lo_freq in lo_freq_range:
            platform.qubits[qubit].drive.lo_frequency = lo_freq
            results = platform.sweep(sequence, options, bias_sweeper, freq_sweeper)

            result = results[ro_pulses[qubit].serial]
            sweetspot = qubits[qubit].sweetspot
            data.register_qubit(
                qubit,
                lo_freq,
                frequencies=delta_frequency_range + qd_pulses[qubit].frequency,
                biases=delta_bias_range + sweetspot,
                msr_matrix=result.magnitude,
                phase_matrix=result.phase,
            )

    return data


def _fit(data: SpuriousData) -> SpuriousResults:
    return SpuriousResults()


def _plot(data: SpuriousData, fit: SpuriousResults, qubit):
    """Plotting function for Spurious Identification."""
    return None


def _update(results: SpuriousResults, platform: Platform, qubit: QubitId):
    pass


spurious_identification = Routine(_acquisition, _fit, _plot, _update)
"""Spurious Identification Routine object."""
