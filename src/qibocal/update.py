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
    ro_channel = platform.qubits[qubit].probe
    platform.update({f"configs.{ro_channel}.frequency": freq})


def bare_resonator_frequency(freq: float, platform: Platform, qubit: QubitId):
    """Update rbare frequency value in platform for specific qubit."""
    platform.calibration.single_qubits[qubit].resonator.bare_frequency = int(freq)


def dressed_resonator_frequency(freq: float, platform: Platform, qubit: QubitId):
    """Update rbare frequency value in platform for specific qubit."""
    platform.calibration.single_qubits[qubit].resonator.dressed_frequency = int(freq)


def readout_amplitude(amp: float, platform: Platform, qubit: QubitId):
    """Update readout amplitude value in platform for specific qubit."""
    platform.update({f"native_gates.single_qubit.{qubit}.MZ.0.1.probe.amplitude": amp})


def drive_frequency(
    freq: Union[float, tuple, list], platform: Platform, qubit: QubitId
):
    """Update drive frequency value in platform for specific qubit."""
    if isinstance(freq, Iterable):
        freq = freq[0]
    drive_channel = platform.qubits[qubit].drive
    platform.update({f"configs.{drive_channel}.frequency": freq})


def drive_amplitude(amp: Union[float, tuple, list], platform: Platform, qubit: QubitId):
    """Update drive frequency value in platform for specific qubit."""
    if isinstance(amp, Iterable):
        amp = amp[0]
    platform.update({f"native_gates.single_qubit.{qubit}.RX.0.1.amplitude": amp})


def drive_duration(
    duration: Union[int, tuple, list], platform: Platform, qubit: QubitId
):
    """Update drive duration value in platform for specific qubit."""
    if isinstance(duration, Iterable):
        duration = duration[0]
    platform.update(
        {f"native_gates.single_qubit.{qubit}.RX.0.1.duration": int(duration)}
    )


def crosstalk_matrix(
    matrix_element: float, platform: Platform, qubit: QubitId, flux_qubit: QubitId
):
    """Update crosstalk_matrix element."""
    if platform.calibration.flux_crosstalk_matrix is None:
        platform.calibration.flux_crosstalk_matrix = np.zeros(
            (platform.calibration.nqubits, platform.calibration.nqubits)
        )
    platform.calibration.set_crosstalk_element(qubit, flux_qubit, matrix_element)


def iq_angle(angle: float, platform: Platform, qubit: QubitId):
    """Update iq angle value in platform for specific qubit."""
    ro_channel = platform.qubits[qubit].acquisition
    platform.update({f"configs.{ro_channel}.iq_angle": angle})


def threshold(threshold: float, platform: Platform, qubit: QubitId):
    ro_channel = platform.qubits[qubit].acquisition
    platform.update({f"configs.{ro_channel}.threshold": threshold})


def mean_gnd_states(ground_state: list, platform: Platform, qubit: QubitId):
    """Update mean ground state value in platform for specific qubit."""
    platform.calibration.single_qubits[qubit].readout.ground_state = ground_state


def mean_exc_states(excited_state: list, platform: Platform, qubit: QubitId):
    """Update mean excited state value in platform for specific qubit."""
    platform.calibration.single_qubits[qubit].readout.excited_state = excited_state


def readout_fidelity(fidelity: float, platform: Platform, qubit: QubitId):
    """Update fidelity of single shot classification."""
    platform.calibration.single_qubits[qubit].readout.fidelity = float(fidelity)


def virtual_phases(
    phases: dict[QubitId, float], native: str, platform: Platform, pair: QubitPairId
):
    pass
    # """Update virtual phases for given qubits in pair in results."""
    # virtual_z_pulses = {
    #     pulse.qubit.name: pulse
    #     for pulse in getattr(platform.pairs[pair].native_gates, native).pulses
    #     if isinstance(pulse, VirtualZPulse)
    # }
    # for qubit_id, phase in phases.items():
    #     if qubit_id in virtual_z_pulses:
    #         virtual_z_pulses[qubit_id].phase = phase
    #     else:
    #         virtual_z_pulses[qubit_id] = VirtualZPulse(
    #             phase=phase, qubit=platform.qubits[qubit_id]
    #         )
    #         getattr(platform.pairs[pair].native_gates, native).pulses.append(
    #             virtual_z_pulses[qubit_id]
    #         )


def CZ_duration(duration: int, platform: Platform, pair: QubitPairId):
    """Update CZ duration for specific pair."""
    platform.update({f"native_gates.two_qubit.{pair}.CZ.0.1.duration": duration})


def CZ_amplitude(amp: float, platform: Platform, pair: QubitPairId):
    """Update CZ amplitude for specific pair."""
    platform.update({f"native_gates.two_qubit.{pair}.CZ.0.1.amp": amp})


def iSWAP_duration(duration: int, platform: Platform, pair: QubitPairId):
    """Update iSWAP_duration duration for specific pair."""
    platform.update({f"native_gates.two_qubit.{pair}.CZ.0.1.duration": duration})


def iSWAP_amplitude(amp: float, platform: Platform, pair: QubitPairId):
    """Update iSWAP_duration amplitude for specific pair."""
    platform.update({f"native_gates.two_qubit.{pair}.CZ.0.1.amp": amp})


def t1(t1: int, platform: Platform, qubit: QubitId):
    """Update t1 value in platform for specific qubit."""
    platform.calibration.single_qubits[qubit].t1 = tuple(t1)


def t2(t2: int, platform: Platform, qubit: QubitId):
    """Update t2 value in platform for specific qubit."""
    platform.calibration.single_qubits[qubit].t2 = tuple(t2)


def t2_spin_echo(t2_spin_echo: float, platform: Platform, qubit: QubitId):
    """Update t2 echo value in platform for specific qubit."""
    platform.calibration.single_qubits[qubit].t2_spin_echo = tuple(t2_spin_echo)


def drag_pulse_beta(beta: float, platform: Platform, qubit: QubitId):
    """Update beta parameter value in platform for specific qubit."""
    platform.update(
        {
            f"native_gates.single_qubit.{qubit}.RX.0.1.envelope.kind": "drag",
            f"native_gates.single_qubit.{qubit}.RX.0.1.envelope.beta": beta,
        }
    )


def sweetspot(sweetspot: float, platform: Platform, qubit: QubitId):
    """Update sweetspot parameter in platform for specific qubit."""
    platform.calibration.single_qubits[qubit].qubit.sweetspot = float(sweetspot)


def flux_offset(offset: float, platform: Platform, qubit: QubitId):
    """Update flux offset parameter in platform for specific qubit."""
    platform.update({f"configs.{platform.qubits[qubit].flux}.offset": offset})


def frequency_12_transition(frequency: int, platform: Platform, qubit: QubitId):
    channel = platform.qubits[qubit].drive_qudits[1, 2]
    platform.update({f"configs.{channel}.frequency": frequency})
    platform.calibration.single_qubits[qubit].qubit.frequency_12 = int(frequency)


def drive_12_amplitude(amplitude: float, platform: Platform, qubit: QubitId):
    platform.update(
        {f"native_gates.single_qubit.{qubit}.RX12.0.1.amplitude": amplitude}
    )


def drive_12_duration(
    duration: Union[int, tuple, list], platform: Platform, qubit: QubitId
):
    """Update drive duration value in platform for specific qubit."""
    platform.update(
        {f"native_gates.single_qubit.{qubit}.RX12.0.1.duration": int(duration)}
    )


def asymmetry(asymmetry: float, platform: Platform, qubit: QubitId):
    platform.calibration.single_qubits[qubit].asymmetry = float(asymmetry)


def coupling(g: float, platform: Platform, qubit: QubitId):
    platform.calibration.single_qubits[qubit].readout.coupling = float(g)


def kernel(kernel: np.ndarray, platform: Platform, qubit: QubitId):
    ro_channel = platform.qubits[qubit].acquisition
    platform.update({f"configs.{ro_channel}.kernel": kernel})
