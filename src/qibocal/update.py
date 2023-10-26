"""Helper functions to update parameters in platform."""
from typing import Union

from qibolab import pulses
from qibolab.native import VirtualZPulse
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from qibocal.protocols.characterization.utils import GHZ_TO_HZ

CLASSIFICATION_PARAMS = [
    "threshold",
    "iq_angle",
    "mean_gnd_states",
    "mean_exc_states",
    "classifier_hpars",
]


def readout_frequency(freq: float, platform: Platform, qubit: QubitId):
    """Update readout frequency value in platform for specific qubit."""
    mz = platform.qubits[qubit].native_gates.MZ
    freq_hz = int(freq * GHZ_TO_HZ)
    mz.frequency = freq_hz
    if mz.if_frequency is not None:
        mz.if_frequency = freq_hz - platform.qubits[qubit].readout.lo_frequency
    platform.qubits[qubit].readout_frequency = freq_hz


def bare_resonator_frequency(freq: float, platform: Platform, qubit: QubitId):
    """Update rbare frequency value in platform for specific qubit."""
    platform.qubits[qubit].bare_resonator_frequency = int(freq * GHZ_TO_HZ)


def readout_amplitude(amp: float, platform: Platform, qubit: QubitId):
    """Update readout amplitude value in platform for specific qubit."""
    platform.qubits[qubit].native_gates.MZ.amplitude = float(amp)


def readout_attenuation(att: int, platform: Platform, qubit: QubitId):
    """Update readout attenuation value in platform for specific qubit."""
    platform.qubits[qubit].readout.attenuation = int(att)


def drive_frequency(freq: Union[float, tuple], platform: Platform, qubit: QubitId):
    """Update drive frequency value in platform for specific qubit."""
    if isinstance(freq, tuple):
        freq = freq[0]
    freq = int(freq * GHZ_TO_HZ)
    platform.qubits[qubit].native_gates.RX.frequency = int(freq)
    platform.qubits[qubit].drive_frequency = int(freq)


def drive_amplitude(amp: Union[float, tuple], platform: Platform, qubit: QubitId):
    """Update drive frequency value in platform for specific qubit."""
    if isinstance(amp, tuple):
        amp = amp[0]
    platform.qubits[qubit].native_gates.RX.amplitude = float(amp)


def drive_duration(duration: Union[int, tuple], platform: Platform, qubit: QubitId):
    """Update drive duration value in platform for specific qubit."""
    if isinstance(duration, tuple):
        duration = duration[0]
    platform.qubits[qubit].native_gates.RX.duration = int(duration)


def iq_angle(angle: float, platform: Platform, qubit: QubitId):
    """Update iq angle value in platform for specific qubit."""
    platform.qubits[qubit].iq_angle = float(angle)


def threshold(threshold: float, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].threshold = float(threshold)


def mean_gnd_states(gnd_state: list, platform: Platform, qubit: QubitId):
    """Update mean ground state value in platform for specific qubit."""
    platform.qubits[qubit].mean_gnd_states = gnd_state


def mean_exc_states(exc_state: list, platform: Platform, qubit: QubitId):
    """Update mean excited state value in platform for specific qubit."""
    platform.qubits[qubit].mean_exc_states = exc_state


def readout_fidelity(fidelity: float, platform: Platform, qubit: QubitId):
    """Update fidelity of single shot classification."""
    platform.qubits[qubit].readout_fidelity = float(fidelity)


def assignment_fidelity(fidelity: float, platform: Platform, qubit: QubitId):
    """Update fidelity of single shot classification."""
    platform.qubits[qubit].assignment_fidelity = float(fidelity)


def virtual_phases(phases: dict[QubitId, float], platform: Platform, pair: QubitPairId):
    """Update virtual phases for given qubits in pair in results."""
    virtual_z_pulses = {
        pulse.qubit.name: pulse
        for pulse in platform.pairs[pair].native_gates.CZ.pulses
        if isinstance(pulse, VirtualZPulse)
    }
    for qubit_id, phase in phases.items():
        if qubit_id in virtual_z_pulses:
            virtual_z_pulses[qubit_id].phase = phase
        else:
            virtual_z_pulses[qubit_id] = VirtualZPulse(
                phase=phase, qubit=platform.qubits[qubit_id]
            )
            platform.pairs[pair].native_gates.CZ.pulses.append(
                virtual_z_pulses[qubit_id]
            )


def CZ_duration(duration: int, platform: Platform, pair: QubitPairId):
    """Update CZ duration for specific pair."""
    if pair not in platform.pairs:
        pair = (pair[1], pair[0])
    for pulse in platform.pairs[pair].native_gates.CZ.pulses:
        if pulse.qubit.name == pair[1]:
            pulse.duration = int(duration)


def CZ_amplitude(amp: float, platform: Platform, pair: QubitPairId):
    """Update CZ amplitude for specific pair."""
    if pair not in platform.pairs:
        pair = (pair[1], pair[0])
    for pulse in platform.pairs[pair].native_gates.CZ.pulses:
        if pulse.qubit.name == pair[1]:
            pulse.amplitude = float(amp)


def t1(t1: int, platform: Platform, qubit: QubitId):
    """Update t1 value in platform for specific qubit."""
    if isinstance(t1, tuple):
        platform.qubits[qubit].t1 = int(t1[0])
    else:
        platform.qubits[qubit].t1 = int(t1)


def t2(t2: int, platform: Platform, qubit: QubitId):
    """Update t2 value in platform for specific qubit."""
    if isinstance(t2, tuple):
        platform.qubits[qubit].t2 = int(t2[0])
    else:
        platform.qubits[qubit].t2 = int(t2)


def t2_spin_echo(t2_spin_echo: float, platform: Platform, qubit: QubitId):
    """Update t2 echo value in platform for specific qubit."""
    if isinstance(t2_spin_echo, tuple):
        platform.qubits[qubit].t2_spin_echo = int(t2_spin_echo[0])
    else:
        platform.qubits[qubit].t2_spin_echo = int(t2_spin_echo)


def drag_pulse_beta(beta: float, platform: Platform, qubit: QubitId):
    """Update beta parameter e value in platform for specific qubit."""
    pulse = platform.qubits[qubit].native_gates.RX.pulse(start=0)
    rel_sigma = pulse.shape.rel_sigma
    drag_pulse = pulses.Drag(rel_sigma=rel_sigma, beta=beta)
    platform.qubits[qubit].native_gates.RX.shape = repr(drag_pulse)


def sweetspot(sweetspot: float, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].sweetspot = float(sweetspot)


def frequency_12_transition(frequency: int, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].native_gates.RX12.frequency = int(frequency * GHZ_TO_HZ)


def drive_12_amplitude(amplitude: float, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].native_gates.RX12.amplitude = float(amplitude)


def twpa_frequency(frequency: int, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].twpa.local_oscillator.frequency = int(frequency)


def twpa_power(power: float, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].twpa.local_oscillator.power = float(power)


def anharmonicity(anharmonicity: float, platform: Platform, qubit: QubitId):
    platform.qubits[qubit].anharmonicity = int(anharmonicity * GHZ_TO_HZ)
