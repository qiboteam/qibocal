"""Helper functions to update parameters in platform."""
from typing import Union

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
        platform.qubits[qubit].native_gates.MZ.amplitude = amp


def readout_attenuation(results: dict[QubitId, float], platform: Platform):
    """Update readout attenuation value in platform for each qubit in results."""
    for qubit, att in results.items():
        platform.qubits[qubit].readout.attenuation = att


def drive_frequency(results: dict[QubitId, Union[float, tuple]], platform: Platform):
    """Update drive frequency value in platform for each qubit in results."""
    for qubit, freq in results.items():
        if isinstance(
            freq, tuple
        ):  # TODO: remove this branching after error bars propagation
            freq = int(freq[0] * GHZ_TO_HZ)
        else:
            freq = int(freq * GHZ_TO_HZ)
        platform.qubits[qubit].native_gates.RX.frequency = freq
        platform.qubits[qubit].drive_frequency = freq


def drive_amplitude(results: dict[QubitId, float], platform: Platform):
    """Update drive frequency value in platform for each qubit in results."""
    for qubit, amp in results.items():
        platform.qubits[qubit].native_gates.RX.amplitude = amp


def iq_angle(results: dict[QubitId, float], platform: Platform):
    """Update iq angle value in platform for each qubit in results."""
    for qubit, angle in results.items():
        platform.qubits[qubit].iq_angle = angle


def threshold(results: dict[QubitId, float], platform: Platform):
    """Update threshold value in platform for each qubit in results."""
    for qubit, threshold in results.items():
        platform.qubits[qubit].threshold = threshold


def mean_gnd_states(results: dict[QubitId, float], platform: Platform):
    """Update mean ground state value in platform for each qubit in results."""
    for qubit, gnd_state in results.items():
        platform.qubits[qubit].mean_gnd_states = gnd_state


def mean_exc_states(results: dict[QubitId, float], platform: Platform):
    """Update mean excited state value in platform for each qubit in results."""
    for qubit, exc_state in results.items():
        platform.qubits[qubit].mean_exc_states = exc_state


def classifiers_hpars(results: dict[QubitId, float], platform: Platform):
    """Update classifier hyperparameters in platform for each qubit in results."""
    for qubit, hpars in results.items():
        platform.qubits[qubit].classifiers_hpars = hpars
