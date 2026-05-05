from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Platform,
    Pulse,
    PulseId,
    PulseLike,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitId, Results
from qibocal.protocols.utils import readout_frequency
from qibocal.result import collect, magnitude


@dataclass
class RamseyParameters(Parameters):
    """Ramsey runcard inputs."""

    delay_between_pulses_start: float
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: float
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: float
    """Step delay between RX(pi/2) pulses in ns."""
    detuning: Optional[float] = None
    """Frequency detuning [Hz] (optional).
        If 0 standard Ramsey experiment is performed."""

    @property
    def delay_range(self) -> tuple[float, float, float]:
        """
        Return a tuple with the delay times between pulses.
        """
        return (
            self.delay_between_pulses_start,
            self.delay_between_pulses_end,
            self.delay_between_pulses_step,
        )


@dataclass
class RamseyResults(Results):
    """Ramsey outputs."""

    detuning: Optional[float] = None
    """Qubit frequency detuning."""
    frequency: dict[QubitId, Union[float, list[float]]] = field(default_factory=dict)
    """Drive frequency [GHz] for each qubit."""
    t2: dict[QubitId, Union[float, list[float]]] = field(default_factory=dict)
    """T2 for each qubit [ns]."""
    delta_phys: dict[QubitId, Union[float, list[float]]] = field(default_factory=dict)
    """Drive frequency [Hz] correction for each qubit."""
    delta_fitting: dict[QubitId, Union[float, list[float]]] = field(
        default_factory=dict
    )
    """Raw drive frequency [Hz] correction for each qubit.
       including the detuning."""
    fitted_parameters: dict[QubitId, list[float]] = field(default_factory=dict)
    """Raw fitting output."""


@dataclass
class RamseyData(Data):
    """Ramsey acquisition outputs."""

    detuning: Optional[float] = None
    """Frequency detuning [Hz]."""
    qubit_freqs: dict[QubitId, float] = field(default_factory=dict)
    """Qubit freqs for each qubit."""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def waits(self):
        """
        Return a list with the waiting times without repetitions.
        """
        qubit = next(iter(self.data))
        return np.unique(self.data[qubit].wait)

    def compute_qubit_signal(self, qubit: QubitId) -> npt.NDArray[np.float64]:
        """
        Return the signal magnitude for a given qubit.
        """
        return magnitude(collect(self.data[qubit].i, self.data[qubit].q))


def ramsey_sequence(
    platform: Platform,
    targets: list[QubitId],
    wait: int = 0,
    target_qubit: Optional[QubitId] = None,
    flux_pulse_amplitude: Optional[float] = None,
) -> tuple[PulseSequence, list[PulseLike]]:
    """Pulse sequence used in Ramsey (detuned) experiments.

    The pulse sequence is the following:

    RX90 -- wait -- RX90 -- MZ
    """
    delays = 2 * len(targets) * [Delay(duration=wait)]
    if flux_pulse_amplitude is not None:
        flux_pulses = len(targets) * [
            Pulse(duration=0, amplitude=flux_pulse_amplitude, envelope=Rectangular())
        ]
    else:
        flux_pulses = []
    sequence = PulseSequence()
    for i, qubit in enumerate(targets):
        natives = platform.natives.single_qubit[qubit]
        qd_channel = platform.qubits[qubit].drive
        rx90_sequence = natives.R(theta=np.pi / 2)
        ro_channel, ro_pulse = natives.MZ()[0]

        sequence += rx90_sequence
        sequence.append((qd_channel, delays[2 * i]))
        sequence += rx90_sequence
        sequence.extend(
            [
                (ro_channel, Delay(duration=2 * rx90_sequence.duration)),
                (ro_channel, delays[2 * i + 1]),
                (ro_channel, ro_pulse),
            ]
        )
        if flux_pulse_amplitude is not None:
            flux_channel = platform.qubits[qubit].flux
            sequence.append((flux_channel, Delay(duration=rx90_sequence.duration)))
            sequence.append((flux_channel, flux_pulses[i]))
        if target_qubit is not None:
            assert target_qubit not in targets, (
                f"Cannot run Ramsey experiment on qubit {target_qubit} if it is already in Ramsey sequence."
            )
            natives = platform.natives.single_qubit[target_qubit]
            sequence += natives.RX()

    return sequence, delays + flux_pulses


def execute_experiment(
    sequence: PulseSequence,
    delays: list[PulseLike],
    platform: Platform,
    targets: list[QubitId],
    params: RamseyParameters,
    return_probs: bool,
) -> dict[PulseId, Results]:
    """Execute Ramsey experiment on the platform."""

    updates = []
    updates += [
        {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
        for q in targets
    ]
    if params.detuning is not None:
        for qubit in targets:
            channel = platform.qubits[qubit].drive
            f0 = platform.config(channel).frequency
            updates.append({channel: {"frequency": f0 + params.detuning}})

    sweeper = Sweeper(
        parameter=Parameter.duration,
        range=params.delay_range,
        pulses=delays,
    )

    # execute the sweep
    result_obj = (
        AcquisitionType.DISCRIMINATION if return_probs else AcquisitionType.INTEGRATION
    )
    return platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        updates=updates,
        relaxation_time=params.relaxation_time,
        acquisition_type=result_obj,
        averaging_mode=AveragingMode.CYCLIC,
    )
