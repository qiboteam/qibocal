from dataclasses import dataclass, field

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
    pass


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
    detuning: float | None = None
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

    detuning: float | None = None
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

    detuning: float | None = None
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


def single_qubit_ramsey_sequence(
    platform: Platform,
    target: QubitId,
    wait: int = 0,
    flux_pulse_amplitude: float | None = None,
) -> tuple[PulseSequence, PulseSequence, list[Pulse | Delay]]:
    """Pulse sequence used in Ramsey (detuned) experiments.

    The pulse sequence is the following:

    RX90 -- wait -- RX90 -- MZ
    """
    delays = [Delay(duration=wait), Delay(duration=wait)]

    # TODO: move flux lines outside this function
    if flux_pulse_amplitude is not None:
        flux_pulse = [
            Pulse(duration=0, amplitude=flux_pulse_amplitude, envelope=Rectangular())
        ]
    else:
        flux_pulse = []

    sequence = PulseSequence()
    ro_sequence = PulseSequence()
    natives = platform.natives.single_qubit[target]
    qd_channel = platform.qubits[target].drive
    rx90_sequence = natives.R(theta=np.pi / 2)
    ro_channel, ro_pulse = natives.MZ()[0]

    sequence += rx90_sequence
    sequence.append((qd_channel, delays[0]))
    sequence += rx90_sequence

    ro_sequence.extend(
        [
            (ro_channel, Delay(duration=2 * rx90_sequence.duration)),
            (ro_channel, delays[1]),
            (ro_channel, ro_pulse),
        ]
    )

    # TODO: move flux lines outside this function
    if flux_pulse_amplitude is not None:
        flux_channel = platform.qubits[target].flux
        sequence.append((flux_channel, Delay(duration=rx90_sequence.duration)))
        sequence.append((flux_channel, flux_pulse[0]))

    return sequence, ro_sequence, delays + flux_pulse


def ramsey_sequence(
    platform: Platform,
    targets: list[QubitId],
    wait: int = 0,
    flux_pulse_amplitude: float | None = None,
) -> tuple[PulseSequence, PulseSequence, list[Pulse | Delay]]:
    """Pulse sequence used in Ramsey (detuned) experiments.
    To be used to run in parallel multiple Ramsey sequences on a qubit list.

    The pulse sequence is the following:

    RX90 -- wait -- RX90 -- MZ
    """
    full_sequence = PulseSequence()
    full_ro_sequence = PulseSequence()
    full_delays: list[Pulse | Delay] = []
    for qubit in targets:
        qubit_seq, qubit_ro_seq, qubit_delays = single_qubit_ramsey_sequence(
            platform=platform,
            target=qubit,
            wait=wait,
            flux_pulse_amplitude=flux_pulse_amplitude,
        )
        full_sequence += qubit_seq
        full_ro_sequence += qubit_ro_seq
        full_delays += qubit_delays

    return full_sequence, full_ro_sequence, full_delays


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
