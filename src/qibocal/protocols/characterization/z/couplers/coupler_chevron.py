from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Parameters, Results, Routine
from qibocal.protocols.characterization.two_qubit_interaction.chevron import (
    ChevronData,
    _plot,
)
from qibocal.protocols.characterization.two_qubit_interaction.utils import order_pair


@dataclass
class ChevronCouplersParameters(Parameters):
    amplitude_min: float
    """Amplitude minimum."""
    amplitude_max: float
    """Amplitude maximum."""
    amplitude_step: float
    """Amplitude step."""
    duration_min: float
    """Duration minimum."""
    duration_max: float
    """Duration maximum."""
    duration_step: float
    """Duration step."""
    dt: Optional[int] = 0
    """Time delay between flux pulses and readout."""
    # parking: bool = True
    # """Wether to park non interacting qubits or not."""

    native_gate: Optional[str] = "CZ"
    """Native gate to implement, CZ or iSWAP."""

    """ChevronCouplers protocol parameters.

    Amplitude and duration are referred to the coupler pulse.
    """


@dataclass
class ChevronCouplersData(ChevronData):
    """Data structure for chevron couplers protocol."""

    native_amplitude: dict[QubitPairId, np.float64] = field(default_factory=dict)


def _aquisition(
    params: ChevronCouplersParameters,
    platform: Platform,
    qubits: list[QubitPairId],
) -> ChevronCouplersData:
    r"""
    Perform an CZ experiment between pairs of qubits by changing the coupler state,
    qubits need to be pulses into their interaction point.

    Args:
        platform: Platform to use.
        params: Experiment parameters.
        qubits (list): List of pairs to use sequentially.

    Returns:
        ChevronCouplersData: Acquisition data.
    """
    # define the parameter to sweep and its range:
    delta_amplitude_range = np.arange(
        params.amplitude_min,
        params.amplitude_max,
        params.amplitude_step,
    )
    delta_duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    # create a DataUnits object to store the results,
    data = ChevronCouplersData()
    # sort high and low frequency qubit
    for pair in qubits:
        sequence = PulseSequence()
        ordered_pair = order_pair(pair, platform.qubits)

        # initialize in system in 11(CZ) or 10(iSWAP) state
        if params.native_gate == "CZ":
            initialize_lowfreq = platform.create_RX_pulse(ordered_pair[0], start=0)
            sequence.add(initialize_lowfreq)

        initialize_highfreq = platform.create_RX_pulse(ordered_pair[1], start=0)

        sequence.add(initialize_highfreq)

        if params.native_gate == "CZ":
            native_gate, _ = platform.create_CZ_pulse_sequence(
                (ordered_pair[1], ordered_pair[0]),
                start=sequence.finish + params.dt,
            )
        elif params.native_gate == "iSWAP":
            native_gate, _ = platform.create_iSWAP_pulse_sequence(
                (ordered_pair[1], ordered_pair[0]),
                start=sequence.finish + params.dt,
            )

        data.native_amplitude[ordered_pair] = getattr(
            native_gate.coupler_pulses(*pair)[:1][0], "amplitude"
        )
        sequence.add(native_gate)

        ro_pulse1 = platform.create_MZ_pulse(
            ordered_pair[1], start=sequence.finish + params.dt
        )
        ro_pulse2 = platform.create_MZ_pulse(
            ordered_pair[0], start=sequence.finish + params.dt
        )

        sequence += ro_pulse1 + ro_pulse2

        sweeper_amplitude = Sweeper(
            Parameter.amplitude,
            delta_amplitude_range,
            pulses=[p for p in native_gate.coupler_pulses(*pair)][:1],
            type=SweeperType.ABSOLUTE,
        )
        sweeper_duration = Sweeper(
            Parameter.duration,
            delta_duration_range,
            pulses=[p for p in native_gate.coupler_pulses(*pair)],
        )

        # repeat the experiment as many times as defined by nshots
        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.SINGLESHOT,
            ),
            sweeper_duration,
            sweeper_amplitude,
        )

        # TODO: classify shots
        classification_parameters = {}
        classification_parameters[0] = {}
        classification_parameters[0][0] = ()
        classification_parameters[0][1] = ()
        classification_parameters[0][2] = ()
        classification_parameters[1] = {}
        classification_parameters[1][0] = ()
        classification_parameters[1][1] = ()
        classification_parameters[1][2] = ()
        classification_parameters[2] = {}
        classification_parameters[2][0] = ()
        classification_parameters[2][1] = ()
        classification_parameters[2][2] = ()
        classification_parameters[3] = {}
        classification_parameters[3][0] = ()
        classification_parameters[3][1] = ()
        classification_parameters[3][2] = ()
        classification_parameters[4] = {}
        classification_parameters[4][0] = ()
        classification_parameters[4][1] = ()
        classification_parameters[4][2] = ()

        def classify():
            return (1, 0, 0)  # (prob state 0, prob state 1, prob state 2) RGB

        data.register_qubit(
            ordered_pair[0],
            ordered_pair[1],
            delta_duration_range,
            delta_amplitude_range,
            results[ordered_pair[0]].probability(state=1),
            results[ordered_pair[1]].probability(state=1),
        )

    return data


@dataclass
class ChevronCouplersResults(Results):
    """Empty fitting outputs for chevron couplers is not implemented in this case."""


def _fit(data: ChevronCouplersData) -> ChevronCouplersResults:
    """ "Results for ChevronCouplers."""
    return ChevronCouplersResults()


def plot(data: ChevronCouplersData, fit: ChevronCouplersResults, qubit):
    return _plot(data, None, qubit)


coupler_chevron = Routine(_aquisition, _fit, plot, two_qubit_gates=True)
"""Coupler cz/swap flux routine."""
