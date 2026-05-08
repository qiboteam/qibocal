from dataclasses import dataclass, field
from typing import Optional

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


class InputError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


@dataclass
class RamseyParameters(Parameters):
    """Ramsey runcard inputs."""

    delay: tuple[float, float, float] | None = None
    """Tuple of the sweeper parameters in the form: (start, stop, step)."""
    delay_between_pulses_start: float | None = None
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: float | None = None
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: float | None = None
    """Step delay between RX(pi/2) pulses in ns."""
    detuning: Optional[float] = None
    """Frequency detuning [Hz] (optional).
        If 0 standard Ramsey experiment is performed."""

    @property
    def delay_range(self) -> tuple[float, float, float]:
        """
        Return a tuple with the delay times between pulses.
        """
        if self.delay is None:
            return (
                self.delay_between_pulses_start,
                self.delay_between_pulses_end,
                self.delay_between_pulses_step,
            )

        return self.delay

    def __post_init__(self):
        if any([d is None for d in self.delay_range]):
            raise InputError("Valid delay range not inserted.")


@dataclass
class RamseyResults(Results):
    """Ramsey outputs."""

    detuning: Optional[float] = None
    """Qubit frequency detuning."""
    frequency: dict[QubitId, list[float]] = field(default_factory=dict)
    """Drive frequency [GHz] for each qubit."""
    t2: dict[QubitId, list[float]] = field(default_factory=dict)
    """T2 for each qubit [ns]."""
    delta_phys: dict[QubitId, list[float]] = field(default_factory=dict)
    """Drive frequency [Hz] correction for each qubit."""
    delta_fitting: dict[QubitId, list[float]] = field(default_factory=dict)
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
    def waits(self) -> npt.NDArray:
        """
        Return a list with the waiting times without repetitions.
        """
        qubit = next(iter(self.data))
        return np.unique(self.data[qubit].wait)


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

    flux_pulses = (
        len(targets)
        * [Pulse(duration=wait, amplitude=flux_pulse_amplitude, envelope=Rectangular())]
        if flux_pulse_amplitude is not None
        else []
    )

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
