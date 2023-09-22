"""Helper functions to update parameters in platform."""
from typing import Union

from qibolab.native import VirtualZPulse
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.protocols.characterization.utils import GHZ_TO_HZ

CLASSIFICATION_PARAMS = [
    "threshold",
    "iq_angle",
    "mean_gnd_states",
    "mean_exc_states",
    "classifier_hpars",
]

# TODO: cast all return types


def readout_frequency(results: dict[QubitId, float], platform: Platform):
    """Update readout frequency value in platform for each qubit in results."""
    for qubit, freq in results.items():
        mz = platform.qubits[qubit].native_gates.MZ
        freq_hz = int(freq * GHZ_TO_HZ)
        mz.frequency = freq_hz
        if mz.if_frequency is not None:
            mz.if_frequency = freq_hz - platform.qubits[qubit].readout.lo_frequency
        platform.qubits[qubit].readout_frequency = freq_hz


def bare_resonator_frequency(results: dict[QubitId, float], platform: Platform):
    """Update bare frequency value in platform for each qubit in results."""
    for qubit, freq in results.items():
        platform.qubits[qubit].bare_resonator_frequency = int(freq * GHZ_TO_HZ)


def readout_amplitude(results: dict[QubitId, float], platform: Platform):
    """Update readout amplitude value in platform for each qubit in results."""
    for qubit, amp in results.items():
        platform.qubits[qubit].native_gates.MZ.amplitude = float(amp)


def readout_attenuation(results: dict[QubitId, float], platform: Platform):
    """Update readout attenuation value in platform for each qubit in results."""
    for qubit, att in results.items():
        platform.qubits[qubit].readout.attenuation = int(att)


def drive_frequency(results: dict[QubitId, Union[float, tuple]], platform: Platform):
    """Update drive frequency value in platform for each qubit in results."""
    for qubit, freq in results.items():
        if isinstance(
            freq, tuple
        ):  # TODO: remove this branching after error bars propagation
            freq = int(freq[0] * GHZ_TO_HZ)
        else:
            freq = int(freq * GHZ_TO_HZ)
        platform.qubits[qubit].native_gates.RX.frequency = int(freq)
        platform.qubits[qubit].drive_frequency = int(freq)


def drive_amplitude(results: dict[QubitId, float], platform: Platform):
    """Update drive frequency value in platform for each qubit in results."""
    for qubit, amp in results.items():
        platform.qubits[qubit].native_gates.RX.amplitude = float(amp)


def drive_duration(results: dict[QubitId, float], platform: Platform):
    """Update drive duration value in platform for each qubit in results."""
    for qubit, duration in results.items():
        platform.qubits[qubit].native_gates.RX.duration = int(duration)


def iq_angle(results: dict[QubitId, float], platform: Platform):
    """Update iq angle value in platform for each qubit in results."""
    for qubit, angle in results.items():
        platform.qubits[qubit].iq_angle = float(angle)


def threshold(results: dict[QubitId, float], platform: Platform):
    """Update threshold value in platform for each qubit in results."""
    for qubit, threshold in results.items():
        platform.qubits[qubit].threshold = float(threshold)


def mean_gnd_states(results: dict[QubitId, float], platform: Platform):
    """Update mean ground state value in platform for each qubit in results."""
    for qubit, gnd_state in results.items():
        platform.qubits[qubit].mean_gnd_states = [float(state) for state in gnd_state]


def mean_exc_states(results: dict[QubitId, float], platform: Platform):
    """Update mean excited state value in platform for each qubit in results."""
    for qubit, exc_state in results.items():
        platform.qubits[qubit].mean_exc_states = [float(state) for state in exc_state]


def classifiers_hpars(results: dict[QubitId, float], platform: Platform):
    """Update classifier hyperparameters in platform for each qubit in results."""
    for qubit, hpars in results.items():
        platform.qubits[qubit].classifiers_hpars = hpars


def virtual_phases(
    results: dict[tuple[QubitId, QubitId], dict[QubitId, float]], platform: Platform
):
    """Update virtual phases for given qubits in pair in results."""
    for pair, phases in results.items():
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


def CZ_duration(results: dict[tuple[QubitId, QubitId], int], platform: Platform):
    """Update CZ duration for each pair in results."""
    for pair, duration in results.items():
        for pulse in platform.pairs[pair].native_gates.CZ.pulses:
            if pulse.qubit.name == pair[1]:
                pulse.duration = int(duration)


def CZ_amplitude(results: dict[tuple[QubitId, QubitId], float], platform: Platform):
    """Update CZ amplitude for each pair in results."""
    for pair, amp in results.items():
        for pulse in platform.pairs[pair].native_gates.CZ.pulses:
            if pulse.qubit.name == pair[1]:
                pulse.amplitude = float(amp)


def t1(results: dict[QubitId, int], platform: Platform):
    """Update mean excited state value in platform for each qubit in results."""
    for qubit, t1 in results.items():
        platform.qubits[qubit].t1 = int(t1)


def t2(results: dict[QubitId, int], platform: Platform):
    """Update mean excited state value in platform for each qubit in results."""
    for qubit, t2 in results.items():
        platform.qubits[qubit].t2 = int(t2)


def t2_spin_echo(results: dict[QubitId, float], platform: Platform):
    """Update mean excited state value in platform for each qubit in results."""
    for qubit, t2_spin_echo in results.items():
        platform.qubits[qubit].t2_spin_echo = int(t2_spin_echo)
