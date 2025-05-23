"""Helper functions to update parameters in platform."""

from collections.abc import Iterable
from typing import Union

import numpy as np
from pydantic import BaseModel
from qibolab import Platform, PulseSequence, VirtualZ

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


def drive_amplitude(
    amp: Union[float, tuple, list], rx90: bool, platform: Platform, qubit: QubitId
):
    """Update drive frequency value in platform for specific qubit."""
    if isinstance(amp, Iterable):
        amp = amp[0]
    if rx90:
        platform.update({f"native_gates.single_qubit.{qubit}.RX90.0.1.amplitude": amp})
    else:
        platform.update({f"native_gates.single_qubit.{qubit}.RX.0.1.amplitude": amp})


def drive_duration(
    duration: Union[int, tuple, list], rx90: bool, platform: Platform, qubit: QubitId
):
    """Update drive duration value in platform for specific qubit."""
    if isinstance(duration, Iterable):
        duration = duration[0]
    if rx90:
        platform.update(
            {f"native_gates.single_qubit.{qubit}.RX90.0.1.duration": int(duration)}
        )
    else:
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
    native_sequence = getattr(platform.natives.two_qubit[pair], native)
    new_native = PulseSequence()
    if len(native_sequence) > 1:
        new_native.append(native_sequence[0])
    else:  # pragma: no cover
        new_native = native_sequence
    for qubit, phase in phases.items():
        new_native.append((platform.qubits[qubit].drive, VirtualZ(phase=phase)))

    platform.update(
        {f"native_gates.two_qubit.{f'{pair[0]}-{pair[1]}'}.{native}": new_native}
    )


def CZ_duration(duration: int, platform: Platform, pair: QubitPairId):
    """Update CZ duration for specific pair."""
    platform.update(
        {f"native_gates.two_qubit.{f'{pair[0]}-{pair[1]}'}.CZ.0.1.duration": duration}
    )


def CZ_amplitude(amp: float, platform: Platform, pair: QubitPairId):
    """Update CZ amplitude for specific pair."""
    platform.update(
        {f"native_gates.two_qubit.{f'{pair[0]}-{pair[1]}'}.CZ.0.1.amplitude": amp}
    )


def iSWAP_duration(duration: int, platform: Platform, pair: QubitPairId):
    """Update iSWAP_duration duration for specific pair."""
    platform.update(
        {f"native_gates.two_qubit.{f'{pair[0]}-{pair[1]}'}.CZ.0.1.duration": duration}
    )


def iSWAP_amplitude(amp: float, platform: Platform, pair: QubitPairId):
    """Update iSWAP_duration amplitude for specific pair."""
    platform.update(
        {f"native_gates.two_qubit.{f'{pair[0]}-{pair[1]}'}.CZ.0.1.amplitude": amp}
    )


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


def flux_coefficients(
    flux_coefficients: list[float], platform: Platform, qubit: QubitId
):
    """Update flux-amplitude relation parameters for specific qubit."""
    platform.calibration.single_qubits[qubit].qubit.flux_coefficients = [
        float(value) for value in flux_coefficients
    ]


def flux_offset(offset: float, platform: Platform, qubit: QubitId):
    """Update flux offset parameter in platform for specific qubit."""
    platform.update({f"configs.{platform.qubits[qubit].flux}.offset": offset})


def frequency_12_transition(frequency: int, platform: Platform, qubit: QubitId):
    channel = platform.qubits[qubit].drive_extra[1, 2]
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


def coupling(g: float, platform: Platform, qubit: QubitId):
    platform.calibration.single_qubits[qubit].readout.coupling = float(g)


def kernel(kernel: np.ndarray, platform: Platform, qubit: QubitId):
    ro_channel = platform.qubits[qubit].acquisition
    platform.update({f"configs.{ro_channel}.kernel": kernel})


def feedback(feedback: list[float], platform: Platform, qubit: QubitId):
    """Update flux pulse feedback filter parameter in platform for specific qubit."""
    feedbackQM = feedback.copy()
    feedbackQM = [-feedbackQM[1]]
    platform.update(
        {f"configs.{platform.qubits[qubit].flux}.filter.feedback": feedbackQM}
    )


def feedforward(feedforward: list[float], platform: Platform, qubit: QubitId):
    """Update flux pulse feedforward parameter in platform for specific qubit."""
    platform.update(
        {f"configs.{platform.qubits[qubit].flux}.filter.feedforward": feedforward}
    )
