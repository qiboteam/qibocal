from typing import Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Routine

from ..two_qubit_interaction.utils import order_pair
from .coupler_resonator_spectroscopy import _fit, _plot, _update
from .utils import CouplerSpectroscopyData, CouplerSpectroscopyParameters


class CouplerSpectroscopyParametersQubit(CouplerSpectroscopyParameters):
    drive_duration: Optional[int] = 2000
    """Drive pulse duration to excite the qubit before the measurement"""


def _acquisition(
    params: CouplerSpectroscopyParametersQubit,
    platform: Platform,
    targets: list[QubitPairId],
) -> CouplerSpectroscopyData:
    """
    Data acquisition for CouplerQubit spectroscopy.

    This consist on a frequency sweep on the qubit frequency while we change the flux coupler pulse amplitude of
    the coupler pulse. We expect to enable the coupler during the amplitude sweep and detect an avoided crossing
    that will be followed by the frequency sweep. This needs the qubits at resonance, the routine assumes a sweetspot
    value for the higher frequency qubit that moves it to the lower frequency qubit instead of trying to calibrate both pulses at once. This should be run after
    qubit_spectroscopy to further adjust the coupler sweetspot if needed and get some information
    on the flux coupler pulse amplitude requiered to enable 2q interactions.

    """

    # TODO: Do we  want to measure both qubits on the pair ?

    # create a sequence of pulses for the experiment:
    # Coupler pulse while Drive pulse - MZ

    if params.measured_qubits is None:
        params.measured_qubits = [order_pair(pair, platform)[0] for pair in targets]

    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    offset = {}
    couplers = []
    for i, pair in enumerate(targets):
        ordered_pair = order_pair(pair, platform)
        measured_qubit = params.measured_qubits[i]

        qubit = platform.qubits[measured_qubit].name
        offset[qubit] = platform.pairs[tuple(sorted(ordered_pair))].coupler.sweetspot
        coupler = platform.pairs[tuple(sorted(ordered_pair))].coupler.name
        couplers.append(coupler)

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=params.drive_duration
        )
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )
        if params.amplitude is not None:
            qd_pulses[qubit].amplitude = params.amplitude

        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    sweeper_freq = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in params.measured_qubits],
        type=SweeperType.OFFSET,
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    sweepers = [
        Sweeper(
            Parameter.bias,
            delta_bias_range,
            qubits=couplers,
            type=SweeperType.OFFSET,
        )
    ]

    data = CouplerSpectroscopyData(
        resonator_type=platform.resonator_type,
        offset=offset,
    )

    for bias_sweeper in sweepers:
        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            bias_sweeper,
            sweeper_freq,
        )

    # retrieve the results for every qubit
    for i, pair in enumerate(targets):
        # TODO: May measure both qubits on the pair
        qubit = platform.qubits[params.measured_qubits[i]].name
        result = results[ro_pulses[qubit].serial]
        # store the results
        data.register_qubit(
            qubit,
            signal=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + qd_pulses[qubit].frequency,
            bias=delta_bias_range,
        )
    return data


coupler_qubit_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""CouplerQubitSpectroscopy Routine object."""
