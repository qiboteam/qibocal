import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Qubits, Routine

from ..two_qubit_interaction.chevron import ChevronData, ChevronParameters, _fit, _plot
from ..two_qubit_interaction.utils import order_pair


def _aquisition(
    params: ChevronParameters,
    platform: Platform,
    qubits: Qubits,
) -> ChevronData:
    r"""
    Routine to find the optimal flux pulse amplitude and duration for a CZ/iSWAP gate.

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

        # TODO: Is this the best way to assume you have a 2q gate on the runcard
        # instead of using platform.create_flux_pulse and platform.create_coupler_pulse ?
        if params.native_gate == "CZ":
            native_gate, virtual_z_phase = platform.create_CZ_pulse_sequence(
                (ordered_pair[1], ordered_pair[0]),
                start=0,
            )
        elif params.native_gate == "iSWAP":
            native_gate, virtual_z_phase = platform.create_iSWAP_pulse_sequence(
                (ordered_pair[1], ordered_pair[0]),
                start=0,
            )

        fq_pulse = native_gate[1]
        fx_pulse = native_gate[0]

        fq_pulse.start = sequence.finish + params.dt
        fx_pulse.start = sequence.finish + params.dt

        sequence += fq_pulse
        sequence += fx_pulse

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
            pulses=[fx_pulse],
        )
        sweeper_duration = Sweeper(
            Parameter.duration,
            delta_duration_range,
            pulses=[fx_pulse],
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
            # coupler, # Do we need the coupler here to extract some info ?
            delta_duration_range,
            delta_amplitude_range,
            results[ordered_pair[0]].magnitude,
            results[ordered_pair[1]].magnitude,
        )

    return data


coupler_chevron = Routine(_aquisition, _fit, _plot, two_qubit_gates=True)
"""Coupler cz/swap flux routine."""
