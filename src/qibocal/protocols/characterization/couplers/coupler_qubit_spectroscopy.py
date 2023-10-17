from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from ..flux_dependence.utils import create_data_array, flux_dependence_plot
from ..two_qubit_interaction.utils import order_pair


@dataclass
class CouplerQubitSpectroscopyParameters(Parameters):
    """CouplerQubitSpectroscopy runcard inputs."""

    offset_width: int
    """Width for offset (V)."""
    offset_step: int
    """Frequency step for offset sweep (V)."""
    freq_width: int
    """Width for frequency sweep relative  to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for frequency sweep (Hz)."""
    measured_qubit: QubitId
    """Qubit to readout from the pair"""
    drive_amplitude: Optional[float] = None
    """Readout amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class CouplerQubitSpectroscopyResults(Results):
    """CouplerQubitSpectroscopy outputs."""

    # sweetspot: dict[QubitId, float]
    # """Sweetspot for each coupler."""
    frequency: dict[QubitId, float] = field(metadata=dict(update="readout_frequency"))
    """Readout frequency [GHz] for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitted parameters."""


CouplerSpecType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for coupler resonator spectroscopy."""


@dataclass
class CouplerQubitSpectroscopyData(Data):
    """Data structure for CouplerQubit spectroscopy."""

    """Resonator type."""
    resonator_type: str

    data: dict[QubitId, npt.NDArray[CouplerSpecType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, msr, phase):
        """Store output for single qubit."""
        self.data[qubit] = create_data_array(
            freq, bias, msr, phase, dtype=CouplerSpecType
        )


def _acquisition(
    params: CouplerQubitSpectroscopyParameters, platform: Platform, qubits: Qubits
) -> CouplerQubitSpectroscopyData:
    """Data acquisition for CouplerQubit spectroscopy."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    coupler_pulses = {}

    for pair in qubits:
        # TODO: DO general
        qubit = platform.qubits[params.measured_qubit].name
        # TODO: Qubit pair patch
        ordered_pair = order_pair(pair, platform.qubits)
        coupler = platform.pairs[tuple(sorted(ordered_pair))].coupler

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=1000)
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=2000
        )
        # TODO: This is not being used for now
        coupler_pulses[coupler.name] = platform.create_coupler_pulse(
            coupler, start=0, duration=ro_pulses[qubit].duration + 2000, amplitude=1
        )
        if params.drive_amplitude is not None:
            qd_pulses[qubit].amplitude = params.drive_amplitude

        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])
        sequence.add(coupler_pulses[coupler.name])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    # TODO: fix loop
    sweeper_freq = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in [params.measured_qubit]],
        type=SweeperType.OFFSET,
    )

    # define the parameter to sweep and its range:
    delta_offset_range = np.arange(
        -params.offset_width / 2, params.offset_width / 2, params.offset_step
    )

    # TODO: fix loop
    sweeper_offset = Sweeper(
        Parameter.bias,
        delta_offset_range,
        couplers=[coupler],
        type=SweeperType.ABSOLUTE,
    )

    data = CouplerQubitSpectroscopyData(
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
        sweeper_offset,
        sweeper_freq,
        # sweeper_offset,
    )

    # TODO: fix loop
    # retrieve the results for every qubit
    for pair in qubits:
        # TODO: DO general
        qubit = platform.qubits[params.measured_qubit].name
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        # store the results
        data.register_qubit(
            qubit,
            msr=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + qd_pulses[qubit].frequency,
            bias=delta_offset_range,
        )
    return data


def _fit(data: CouplerQubitSpectroscopyData) -> CouplerQubitSpectroscopyResults:
    """Post-processing function for CouplerQubitSpectroscopy."""
    qubits = data.qubits
    frequency = {}
    fitted_parameters = {}

    for qubit in qubits:
        # TODO: Fix fit
        # freq, fitted_params = lorentzian_fit(
        #     data[qubit], resonator_type=data.resonator_type, fit="resonator"
        # )

        frequency[qubit] = 0
        fitted_parameters[qubit] = {}

    return CouplerQubitSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
    )


def _plot(
    data: CouplerQubitSpectroscopyData,
    qubit,
    fit: CouplerQubitSpectroscopyResults,
):
    """Plotting function for CouplerQubitSpectroscopy."""
    # TODO: fix loop
    qubit = 1
    return flux_dependence_plot(data, fit, qubit)


def _update(
    results: CouplerQubitSpectroscopyResults, platform: Platform, qubit: QubitId
):
    if 1 == 0:
        update.readout_frequency(results.frequency[qubit], platform, qubit)


coupler_qubit_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""CouplerQubitSpectroscopy Routine object."""
