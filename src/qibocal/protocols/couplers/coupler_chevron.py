import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence, PulseType
from qibolab.qubits import QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Results, Routine

from ..two_qubit_interaction.chevron.chevron import (
    ChevronData,
    ChevronParameters,
    _plot,
)
from ..two_qubit_interaction.utils import order_pair


def _acquisition(
    params: ChevronParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> ChevronData:
    r"""
    Perform an CZ experiment between pairs of qubits by changing the coupler state,
    qubits need to be pulses into their interaction point.

    Args:
        platform: Platform to use.
        params: Experiment parameters.
        targets (list): List of pairs to use sequentially.

    Returns:
        ChevronData: Acquisition data.
    """
    # define the parameter to sweep and its range:
    delta_amplitude_range = np.arange(
        params.amplitude_min_factor,
        params.amplitude_max_factor,
        params.amplitude_step_factor,
    )
    delta_duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    data = ChevronData()
    for pair in targets:
        sequence = PulseSequence()

        ordered_pair = order_pair(pair, platform)

        # initialize system to state 11(CZ) or 10(iSWAP)
        if params.native == "CZ":
            initialize_lowfreq = platform.create_RX_pulse(ordered_pair[0], start=0)
            sequence.add(initialize_lowfreq)

        initialize_highfreq = platform.create_RX_pulse(ordered_pair[1], start=0)

        sequence.add(initialize_highfreq)

        if params.native == "CZ":
            native_gate, _ = platform.create_CZ_pulse_sequence(
                (ordered_pair[1], ordered_pair[0]),
                start=sequence.finish + params.dt,
            )
        elif params.native == "iSWAP":
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

        coupler_flux_pulses = [p for p in native_gate.coupler_pulses(*pair)]
        assert (
            len(coupler_flux_pulses) == 1
        ), f"coupler_chevron expects exactly one coupler flux pulse, but {len(coupler_flux_pulses)} are present."
        qubit_flux_pulses = [
            p for p in native_gate.get_qubit_pulses(*pair) if p.type is PulseType.FLUX
        ]
        assert all(
            len(list(filter(lambda x: x.qubit == q, qubit_flux_pulses))) < 2
            for q in pair
        ), f"coupler_chevron expects no more than 1 flux pulse for each qubit, but more are present for the pair {pair}"
        sweeper_amplitude = Sweeper(
            Parameter.amplitude,
            delta_amplitude_range,
            pulses=coupler_flux_pulses,
            type=SweeperType.FACTOR,
        )
        sweeper_duration = Sweeper(
            Parameter.duration,
            delta_duration_range,
            pulses=coupler_flux_pulses + qubit_flux_pulses,
        )

        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            sweeper_duration,
            sweeper_amplitude,
        )

        data.register_qubit(
            ordered_pair[0],
            ordered_pair[1],
            delta_duration_range,
            delta_amplitude_range * data.native_amplitude[ordered_pair],
            results[ordered_pair[0]].probability(state=1),
            results[ordered_pair[1]].probability(state=1),
        )
    data.label = "Probability of state |1>"

    return data


def _fit(data: ChevronData) -> Results:
    """Results for ChevronCouplers."""
    return Results()


def plot(data: ChevronData, fit: Results, target):
    return _plot(data, None, target)


coupler_chevron = Routine(_acquisition, _fit, plot, two_qubit_gates=True)
"""Coupler cz/swap flux routine."""
