import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Qubits, Routine

from ..two_qubit_interaction.utils import order_pair
from .coupler_resonator_spectroscopy import _fit, _plot, _update
from .utils import CouplerSpectroscopyData, CouplerSpectroscopyParameters


def _acquisition(
    params: CouplerSpectroscopyParameters, platform: Platform, qubits: Qubits
) -> CouplerSpectroscopyData:
    """Data acquisition for CouplerQubit spectroscopy."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}

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
        if params.amplitude is not None:
            qd_pulses[qubit].amplitude = params.amplitude

        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

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
    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )

    # TODO: fix loop
    """This sweeper is implemented in the flux pulse amplitude and we need it to be that way. """
    sweeper_bias = Sweeper(
        Parameter.bias,
        delta_bias_range,
        couplers=[coupler],
        type=SweeperType.ABSOLUTE,
    )

    data = CouplerSpectroscopyData(
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
        sweeper_bias,
        sweeper_freq,
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
            bias=delta_bias_range,
        )
    return data


coupler_qubit_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""CouplerQubitSpectroscopy Routine object."""