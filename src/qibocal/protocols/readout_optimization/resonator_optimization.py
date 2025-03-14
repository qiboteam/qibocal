from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import readout_frequency
from qibocal.update import replace


@dataclass
class ResonatorOptimizationParameters(Parameters):
    """Resonator optimization runcard inputs"""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    amplitude_start: float
    """Minimum amplitude multiplicative factor."""
    amplitude_stop: float
    """Maximum amplitude multiplicative factor."""
    amplitude_step: float
    """Step amplitude multiplicative factor."""
    error_threshold: float = 0.003
    """Probability error threshold to stop the best amplitude search"""


@dataclass
class ResonatorOptimizationResults(Results):
    """Resonator optimization outputs"""

    fidelities: dict[QubitId, list]
    """Assignment fidelities."""
    best_freq: dict[QubitId, float]
    """Resonator Frequency with the highest assignment fidelity."""
    best_amp: dict[QubitId, list]
    """Amplitude with lowest error."""
    best_angle: dict[QubitId, float]
    """IQ angle that maximes assignment fidelity."""
    best_threshold: dict[QubitId, float]
    """Threshold that maximes assignment fidelity."""


ResonatorOptimizationType = np.dtype(
    [
        ("error", np.float64),
        ("frequency", np.float64),
        ("amp", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype readout optimization."""


@dataclass
class ResonatorOptimizationData(Data):
    """Data class for resonator optimization protocol."""

    resonator_type: str
    """Resonator type."""
    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray[ResonatorOptimizationType]] = field(
        default_factory=dict
    )


def _acquisition(
    params: ResonatorOptimizationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorOptimizationData:
    r"""
    Data acquisition for readout optimization.

    Args:
        params (ResonatorFrequencyParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (list): list of target qubits to perform the action
    """
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=readout_frequency(q, platform) + delta_frequency_range,
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]

    data = ResonatorOptimizationData(
        resonator_type=platform.resonator_type,
    )

    amplitudes = {}

    for qubit in targets:
        error = 1
        natives = platform.natives.single_qubit[qubit]

        ro_channel, ro_pulse = natives.MZ()[0]
        new_amp = params.amplitude_start

        while error > params.error_threshold and new_amp <= params.amplitude_stop:

            # da definire dopo aver definito gli sweeper per l'ampiezza
            new_ro = replace(ro_pulse, amplitude=new_amp)
            amplitudes[qubit] = new_ro.probe.amplitude

            sequence_0 = PulseSequence()
            sequence_1 = PulseSequence()

            qd_channel, qd_pulse = natives.RX()[0]

            sequence_1.append((qd_channel, qd_pulse))
            sequence_1.append((ro_channel, Delay(duration=qd_pulse.duration)))
            sequence_1.append((ro_channel, new_ro))

            sequence_0.append((ro_channel, new_ro))

            state0_results = platform.execute(
                [sequence_0],
                [sweepers],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
            )

            state1_results = platform.execute(
                [sequence_1],
                [sweepers],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
            )
            result0 = np.concatenate(state0_results[new_ro.id])
            result1 = np.concatenate(state1_results[new_ro.id])

            iq_values = np.concatenate(result0, result1)
            states = [0] * len(result0) + [1] * len(result1)
            model = QubitFit()
            model.fit(iq_values, np.array(states))
            error = model.probability_error
            data.register_qubit(ResonatorOptimizationType, (qubit), dict())

            new_amp += params.amplitude_step

    data.amplitudes = amplitudes
    return data


def _fit(data: ResonatorOptimizationData) -> ResonatorOptimizationResults:
    qubits = data.qubits
    best_freq = {}
    best_amps = {}
    best_angle = {}
    best_threshold = {}
    highest_fidelity = {}

    return ResonatorOptimizationResults(
        best_amp=best_amps,
        fidelities=highest_fidelity,
        best_freq=best_freq,
        best_angle=best_angle,
        best_threshold=best_threshold,
    )


def _plot(
    data: ResonatorOptimizationData, fit: ResonatorOptimizationResults, target: QubitId
):
    """Plotting function for resonator optimization"""

    figures = []
    fitting_report = None
    # if fit is not None:
    return figures, fitting_report


def _update(
    results: ResonatorOptimizationResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.readout_amplitude(results.best_amp[target], platform, target)
    update.readout_frequency(results.best_freq[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


resonator_optimization = Routine(
    _acquisition,
    _fit,
    _plot,
    _update,
)
"""Resonator optimization Routine object"""
