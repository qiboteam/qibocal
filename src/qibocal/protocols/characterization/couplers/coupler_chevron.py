from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Qubits, Results, Routine

from ..two_qubit_interaction.chevron import ChevronData, ChevronParameters, _plot
from ..two_qubit_interaction.utils import order_pair


@dataclass
class ChevronCouplersParameters(ChevronParameters):
    """ChevronCouplers protocol parameters.

    Amplitude and duration are referred to the coupler pulse.
    """


@dataclass
class ChevronCouplersData(ChevronData):
    """Data structure for chevron couplers protocol."""


def _aquisition(
    params: ChevronCouplersParameters,
    platform: Platform,
    qubits: Qubits,
) -> ChevronData:
    r"""
    Routine to find the optimal coupler flux pulse amplitude and duration for a CZ/iSWAP gate.

    The qubits must be at specific frequencies such that the high frequency qubit
    1 to 2 (CZ) / 0 to 1 (iSWAP) transition is at the same frequency as the low frequency qubit 0 to 1 transition.
    At this avoided crossing, the coupling can be turned on and off by applying a flux pulse on the coupler.
    The amplitude of this flux pluse changes the frequency of the coupler. The
    closer the coupler frequency is to the avoided crossing, the stronger the coupling.
    A strong interaction allows for a faster controlled gate.

    Args:
        platform: Platform to use.
        params: Experiment parameters.
        qubits: Dict of QubitPairs.

    Returns:
        DataUnits: Acquisition data.
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
    data = ChevronData()
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
            pulses=[native_gate.coupler_pulses(*pair)],
        )
        sweeper_duration = Sweeper(
            Parameter.duration,
            delta_duration_range,
            pulses=[native_gate.coupler_pulses(*pair)],
        )

        # repeat the experiment as many times as defined by nshots
        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            sweeper_duration,
            sweeper_amplitude,
        )

        # TODO: Explore probabilities instead of magnitude
        data.register_qubit(
            ordered_pair[0],
            ordered_pair[1],
            delta_duration_range,
            delta_amplitude_range,
            results[ordered_pair[0]].magnitude,
            results[ordered_pair[1]].magnitude,
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
