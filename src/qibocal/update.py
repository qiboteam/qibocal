"""Helper functions to update parameters in platform."""

from collections.abc import Iterable
from typing import Union

import numpy as np
from pydantic import BaseModel
from qibolab import Platform

from qibocal.auto.operation import QubitId, QubitPairId

CLASSIFICATION_PARAMS = [
    "threshold",
    "iq_angle",
    "mean_gnd_states",
    "mean_exc_states",
    "classifier_hpars",
]


def replace(model: BaseModel, **update):
    """Replace interface for pydantic models."""
    return model.model_copy(update=update)


def readout_frequency(freq: float, platform: Platform, qubit: QubitId):
    """Update readout frequency value in platform for specific qubit."""
    channel = platform.qubits[qubit].probe
    platform.parameters.configs[channel] = replace(
        platform.config(channel), frequency=freq
    )


def readout_amplitude(amp: float, platform: Platform, qubit: QubitId):
    """Update readout amplitude value in platform for specific qubit."""
    channel, pulse = platform.natives.single_qubit[qubit].MZ[0]
    new_pulse = replace(pulse, probe=replace(pulse.probe, amplitude=amp))
    platform.natives.single_qubit[qubit].MZ[0] = (channel, new_pulse)


def readout_attenuation(att: int, platform: Platform, qubit: QubitId):
    """Update readout attenuation value in platform for specific qubit."""
    # platform.qubits[qubit].readout.attenuation = int(att)


def drive_frequency(
    freq: Union[float, tuple, list], platform: Platform, qubit: QubitId
):
    """Update drive frequency value in platform for specific qubit."""
    # if isinstance(freq, Iterable):
    #    freq = freq[0]
    # freq = int(freq)
    # platform.qubits[qubit].native_gates.RX.frequency = int(freq)
    # platform.qubits[qubit].drive_frequency = int(freq)


def drive_amplitude(amp: Union[float, tuple, list], platform: Platform, qubit: QubitId):
    """Update drive frequency value in platform for specific qubit."""
    if isinstance(amp, Iterable):
        amp = amp[0]
    channel, pulse = platform.natives.single_qubit[qubit].RX[0]
    new_pulse = replace(pulse, amplitude=amp)
    platform.natives.single_qubit[qubit].RX[0] = (channel, new_pulse)


def drive_duration(
    duration: Union[int, tuple, list], platform: Platform, qubit: QubitId
):
    """Update drive duration value in platform for specific qubit."""
    if isinstance(duration, Iterable):
        duration = duration[0]
    channel, pulse = platform.natives.single_qubit[qubit].RX[0]
    new_pulse = replace(pulse, duration=duration)
    platform.natives.single_qubit[qubit].RX[0] = (channel, new_pulse)


def crosstalk_matrix(
    matrix_element: float, platform: Platform, qubit: QubitId, flux_qubit: QubitId
):
    """Update crosstalk_matrix element."""
    platform.qubits[qubit].crosstalk_matrix[flux_qubit] = float(matrix_element)


def iq_angle(angle: float, platform: Platform, qubit: QubitId):
    """Update classification iq angle value in platform for specific qubit."""
    channel = platform.qubits[qubit].acquisition
    platform.parameters.configs[channel] = replace(
        platform.config(channel), iq_angle=angle
    )


def threshold(threshold: float, platform: Platform, qubit: QubitId):
    """Update classification threshold value in platform for specific qubit."""
    channel = platform.qubits[qubit].acquisition
    platform.parameters.configs[channel] = replace(
        platform.config(channel), threshold=threshold
    )


def virtual_phases(
    phases: dict[QubitId, float], native: str, platform: Platform, pair: QubitPairId
):
    """Update virtual phases for given qubits in pair in results."""
    virtual_z_pulses = {
        pulse.qubit.name: pulse
        for pulse in getattr(platform.pairs[pair].native_gates, native).pulses
        if isinstance(pulse, VirtualZPulse)
    }
    for qubit_id, phase in phases.items():
        if qubit_id in virtual_z_pulses:
            virtual_z_pulses[qubit_id].phase = phase
        else:
            virtual_z_pulses[qubit_id] = VirtualZPulse(
                phase=phase, qubit=platform.qubits[qubit_id]
            )
            getattr(platform.pairs[pair].native_gates, native).pulses.append(
                virtual_z_pulses[qubit_id]
            )


def CZ_duration(duration: int, platform: Platform, pair: QubitPairId):
    """Update CZ duration for specific pair."""
    for pulse in platform.pairs[pair].native_gates.CZ.pulses:
        if pulse.qubit.name == pair[1]:
            pulse.duration = int(duration)


def CZ_amplitude(amp: float, platform: Platform, pair: QubitPairId):
    """Update CZ amplitude for specific pair."""
    for pulse in platform.pairs[pair].native_gates.CZ.pulses:
        if pulse.qubit.name == pair[1]:
            pulse.amplitude = float(amp)


def iSWAP_duration(duration: int, platform: Platform, pair: QubitPairId):
    """Update iSWAP_duration duration for specific pair."""
    for pulse in platform.pairs[pair].native_gates.iSWAP.pulses:
        if pulse.qubit.name == pair[1]:
            pulse.duration = int(duration)


def iSWAP_amplitude(amp: float, platform: Platform, pair: QubitPairId):
    """Update iSWAP_duration amplitude for specific pair."""
    for pulse in platform.pairs[pair].native_gates.iSWAP.pulses:
        if pulse.qubit.name == pair[1]:
            pulse.amplitude = float(amp)


def t1(t1: int, platform: Platform, qubit: QubitId):
    """Update t1 value in platform for specific qubit."""
    if isinstance(t1, Iterable):
        platform.qubits[qubit].T1 = int(t1[0])
    else:
        platform.qubits[qubit].T1 = int(t1)


def t2(t2: int, platform: Platform, qubit: QubitId):
    """Update t2 value in platform for specific qubit."""
    if isinstance(t2, Iterable):
        platform.qubits[qubit].T2 = int(t2[0])
    else:
        platform.qubits[qubit].T2 = int(t2)


def t2_spin_echo(t2_spin_echo: float, platform: Platform, qubit: QubitId):
    """Update t2 echo value in platform for specific qubit."""
    if isinstance(t2_spin_echo, Iterable):
        platform.qubits[qubit].T2_spin_echo = int(t2_spin_echo[0])
    else:
        platform.qubits[qubit].T2_spin_echo = int(t2_spin_echo)


def drag_pulse_beta(beta: float, platform: Platform, qubit: QubitId):
    """Update beta parameter value in platform for specific qubit."""
    pulse = platform.qubits[qubit].native_gates.RX.pulse(start=0)
    rel_sigma = pulse.shape.rel_sigma
    drag_pulse = pulses.Drag(rel_sigma=rel_sigma, beta=beta)
    platform.qubits[qubit].native_gates.RX.shape = repr(drag_pulse)


def sweetspot(sweetspot: float, platform: Platform, qubit: QubitId):
    """Update sweetspot parameter in platform for specific qubit."""
    platform.qubits[qubit].sweetspot = float(sweetspot)


def frequency_12_transition(frequency: int, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].native_gates.RX12.frequency = int(frequency)


def drive_12_amplitude(amplitude: float, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].native_gates.RX12.amplitude = float(amplitude)


def drive_12_duration(
    duration: Union[int, tuple, list], platform: Platform, qubit: QubitId
):
    """Update drive duration value in platform for specific qubit."""
    platform.qubits[qubit].native_gates.RX12.duration = int(duration)


def twpa_frequency(frequency: int, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].twpa.local_oscillator.frequency = int(frequency)


def twpa_power(power: float, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].twpa.local_oscillator.power = float(power)


def anharmonicity(anharmonicity: float, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].anharmonicity = int(anharmonicity)


def asymmetry(asymmetry: float, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].asymmetry = float(asymmetry)


def coupling(g: float, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].g = float(g)


def kernel(kernel: np.ndarray, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].kernel = kernel
