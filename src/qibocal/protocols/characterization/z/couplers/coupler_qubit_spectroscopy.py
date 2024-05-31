from dataclasses import dataclass
from typing import Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Parameters, Qubits, Routine
from qibocal.protocols.characterization.two_qubit_interaction.utils import order_pair

from .coupler_resonator_spectroscopy import _fit, _plot, _update
from .utils import CouplerSpectroscopyData


@dataclass
class CouplerSpectroscopyParametersQubitBias(Parameters):
    bias_width: int
    """Width for bias [a.u.]."""
    bias_step: int
    """Frequency step for bias sweep [a.u.]."""
    freq_width: int
    """Width for frequency sweep relative  to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for frequency sweep [Hz]."""
    # TODO: It may be better not to use readout multiplex to avoid readout crosstalk
    measured_qubits: list[QubitId]
    """Qubit to readout from the pair"""
    amplitude: Optional[float] = None
    """Readout or qubit drive amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    readout_delay: Optional[int] = 1000
    """Readout delay before the measurement is done to let the flux coupler pulse act"""
    drive_duration: Optional[int] = 2000
    """Drive pulse duration to excite the qubit before the measurement"""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time [ns]."""


@dataclass
class CouplerSpectroscopyParametersQubitAmplitude(Parameters):
    amplitude_min: float
    """Amplitude minimum."""
    amplitude_max: float
    """Amplitude maximum."""
    amplitude_step: float
    """Amplitude step."""
    freq_width: int
    """Width for frequency sweep relative  to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for frequency sweep [Hz]."""
    # TODO: It may be better not to use readout multiplex to avoid readout crosstalk
    measured_qubits: list[QubitId]
    """Qubit to readout from the pair"""
    amplitude: Optional[float] = None
    """Readout or qubit drive amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    readout_delay: Optional[int] = 1000
    """Readout delay before the measurement is done to let the flux coupler pulse act"""
    drive_duration: Optional[int] = 2000
    """Drive pulse duration to excite the qubit before the measurement"""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time [ns]."""


def _acquisition_bias(
    params: CouplerSpectroscopyParametersQubitBias, platform: Platform, qubits: Qubits
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
    # Different acquisition, for now only measure one and reduce possible crosstalk.

    # create a sequence of pulses for the experiment:
    # Coupler pulse while Drive pulse - MZ

    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    couplers = []
    for i, pair in enumerate(qubits):
        qubit = platform.qubits[params.measured_qubits[i]].name
        # TODO: Qubit pair patch
        ordered_pair = order_pair(pair, platform.qubits)
        couplers.append(platform.pairs[tuple(sorted(ordered_pair))].coupler)

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
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    sweeper_freq = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in params.measured_qubits],
        type=SweeperType.OFFSET,
    )

    # define the parameter to sweep and its range:
    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )

    # This sweeper is implemented in the flux pulse amplitude and we need it to be that way.
    sweeper_bias = Sweeper(
        Parameter.bias,
        delta_bias_range,
        couplers=couplers,
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

    # retrieve the results for every qubit
    for i, pair in enumerate(qubits):
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


def _acquisition_amplitude(
    params: CouplerSpectroscopyParametersQubitAmplitude,
    platform: Platform,
    qubits: Qubits,
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
    # Different acquisition, for now only measure one and reduce possible crosstalk.

    # create a sequence of pulses for the experiment:
    # Coupler pulse while Drive pulse - MZ

    sequence = PulseSequence()
    ro_pulses = {}
    qf_pulses = {}
    qd_pulses = {}
    couplers = []
    for i, pair in enumerate(qubits):
        qubit = platform.qubits[params.measured_qubits[i]].name
        # TODO: Qubit pair patch
        ordered_pair = order_pair(pair, platform.qubits)
        coupler = platform.pairs[tuple(sorted(ordered_pair))].coupler.name
        couplers.append(coupler)

        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )
        if params.amplitude is not None:
            qd_pulses[qubit].amplitude = params.amplitude
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=params.drive_duration
        )
        qf_pulses[coupler] = platform.create_coupler_pulse(
            coupler, 0, ro_pulses[qubit].finish
        )

        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])
        sequence.add(qf_pulses[coupler])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    sweeper_freq = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in params.measured_qubits],
        type=SweeperType.OFFSET,
    )

    # define the parameter to sweep and its range:
    delta_amplitude_range = np.arange(
        params.amplitude_min,
        params.amplitude_max,
        params.amplitude_step,
    )

    # This sweeper is implemented in the flux pulse amplitude and we need it to be that way.
    sweeper_amplitude = Sweeper(
        Parameter.amplitude,
        delta_amplitude_range,
        pulses=[qf_pulses[coupler] for coupler in couplers],
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
        sweeper_amplitude,
        sweeper_freq,
    )

    # retrieve the results for every qubit
    for i, pair in enumerate(qubits):
        # TODO: May measure both qubits on the pair
        qubit = platform.qubits[params.measured_qubits[i]].name
        result = results[ro_pulses[qubit].serial]
        # store the results
        data.register_qubit(
            qubit,
            signal=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + qd_pulses[qubit].frequency,
            bias=delta_amplitude_range,
        )
    return data


coupler_qubit_spectroscopy_bias = Routine(_acquisition_bias, _fit, _plot, _update)
coupler_qubit_spectroscopy_amplitude = Routine(
    _acquisition_amplitude, _fit, _plot, _update
)
"""CouplerQubitSpectroscopy Routine object."""
