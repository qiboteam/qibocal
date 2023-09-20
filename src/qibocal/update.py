"""Helper functions to update parameters in platform."""
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.protocols.characterization.utils import GHZ_TO_HZ


def update_readout_frequency(results: dict[QubitId, float], platform: Platform):
    """Update readout frequency value in platform for each qubit in results."""
    for qubit, freq in results.items():
        mz = platform.qubits[qubit].native_gates.MZ
        freq_hz = int(freq * GHZ_TO_HZ)
        mz.frequency = freq_hz
        if mz.if_frequency is not None:
            mz.if_frequency = freq_hz - platform.get_lo_readout_frequency(qubit)
        platform.qubits[qubit].readout_frequency = freq_hz


def update_bare_resonator_frequency(results: dict[QubitId, float], platform: Platform):
    """Update bare frequency value in platform for each qubit in results."""
    for qubit, freq in results.items():
        platform.qubits[qubit].bare_resonator_frequency = int(freq * GHZ_TO_HZ)


def update_readout_amplitude(results: dict[QubitId, float], platform: Platform):
    """Update readout amplitude value in platform for each qubit in results."""
    for qubit, amp in results.items():
        platform.qubits[qubit].native_gates.MZ.amplitude = amp
