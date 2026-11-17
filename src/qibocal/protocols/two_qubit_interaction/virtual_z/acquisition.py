"""Pulse sequences for the virtual Z phase calibration of two-qubit gates."""

from typing import Literal

import numpy as np
from qibolab import Delay, Pulse, PulseSequence, VirtualZ

from qibocal.calibration import CalibrationPlatform
from qibocal.calibration.calibration import QubitId

from ....update import replace

__all__ = ["create_sequence"]


def _native_flux_pulse(
    platform: CalibrationPlatform,
    native: Literal["CZ", "iSWAP"],
    ordered_pair: tuple[QubitId, QubitId],
    flux_pulse: Pulse | None,
    flux_pulse_max_duration: float | None,
) -> Pulse:
    """Flux pulse of the native gate, played on the high frequency qubit."""
    flux_channel = platform.qubits[ordered_pair[1]].flux
    assert flux_channel is not None, (
        "Flux channel must be defined for the high frequency qubit."
    )
    if flux_pulse is None:
        native_sequence = getattr(platform.natives.two_qubit[ordered_pair], native)
        assert native_sequence is not None, (
            f"{native} native not defined for pair {ordered_pair}."
        )
        flux_pulse_ = next(iter(native_sequence.channel(flux_channel)))
        assert isinstance(flux_pulse_, Pulse), "Flux pulse must be a Pulse object."
        flux_pulse = flux_pulse_

    if flux_pulse_max_duration is not None:
        flux_pulse = replace(flux_pulse, duration=flux_pulse_max_duration)
    return flux_pulse


def _create_cz_sequence(
    platform: CalibrationPlatform,
    setup: Literal["I", "X"],
    oscillating_qubit: QubitId,
    steady_qubit: QubitId,
    ordered_pair: tuple[QubitId, QubitId],
    dt: float,
    flux_pulse_max_duration: float | None = None,
    flux_pulse: Pulse | None = None,
    *,
    vzphase: float = 0.0,
) -> tuple[PulseSequence, Pulse, VirtualZ]:
    """
    Create the pulse sequence for the calibration of CZ virtual phases.

    The sequence includes:
    - A preliminary RX90 pulse on the `oscillating_qubit`.
    - An optional X pulse on the `steady_qubit` based on the `setup` type.
    - A flux pulse implementing the CZ gate.
    - A delay of duration `dt` before the final X90 pulse on the target qubit.
    - An optional X pulse on the `steady_qubit` (leakage detection).
    - Measurement pulses.

    It is possible to specify the maximum duration for the flux pulses with the
    `flux_pulse_max_duration` parameter.

    The function returns:
    - The full experiment pulse sequence.
    - The applied flux pulse.
    - The final `VirtualZ` pulse to be used for phase sweeping.
    """
    oscillating_natives = platform.natives.single_qubit[oscillating_qubit]
    steady_natives = platform.natives.single_qubit[steady_qubit]
    flux_channel = platform.qubits[ordered_pair[1]].flux
    assert flux_channel is not None, (
        "Flux channel must be defined for the high frequency qubit."
    )

    sequence = PulseSequence()
    # X90
    sequence += oscillating_natives.R(theta=np.pi / 2)
    # X
    if setup == "X":
        assert steady_natives.RX is not None
        sequence += steady_natives.RX()

    flux_pulse = _native_flux_pulse(
        platform, "CZ", ordered_pair, flux_pulse, flux_pulse_max_duration
    )

    sequence |= PulseSequence(
        [
            (flux_channel, Delay(duration=dt)),
            (flux_channel, flux_pulse),
            (flux_channel, Delay(duration=dt)),
        ]
    )

    vz = VirtualZ(phase=vzphase)
    sequence |= PulseSequence([(platform.qubits[oscillating_qubit].drive, vz)])

    sequence |= oscillating_natives.R(theta=np.pi / 2)

    if setup == "X":
        sequence += steady_natives.RX()

    sequence |= PulseSequence(
        [
            oscillating_natives.MZ()[0],
            steady_natives.MZ()[0],
        ]
    )

    return sequence, flux_pulse, vz


def _create_iswap_sequence(
    platform: CalibrationPlatform,
    setup: Literal["I", "X"],
    oscillating_qubit: QubitId,
    steady_qubit: QubitId,
    ordered_pair: tuple[QubitId, QubitId],
    dt: float,
    flux_pulse_max_duration: float | None = None,
    flux_pulse: Pulse | None = None,
    *,
    vzphase: float = 0.0,
) -> tuple[PulseSequence, Pulse, VirtualZ]:
    """
    Create the pulse sequence for the calibration of iSWAP virtual phases.

    The iSWAP gate exchanges the excitation between the two qubits, so the roles of
    the qubits before the gate are swapped with respect to
    :func:`create_cz_sequence`. The sequence includes:
    - A preliminary RX90 pulse on the `steady_qubit`, whose superposition is
      transferred to the `oscillating_qubit` by the gate.
    - An optional X pulse on the `oscillating_qubit` based on the `setup` type, whose
      excitation is transferred to the `steady_qubit` by the gate.
    - A flux pulse implementing the iSWAP gate.
    - A delay of duration `dt` before the final X90 pulse on the target qubit.
    - An optional X pulse on the `steady_qubit` (leakage detection).
    - Measurement pulses.

    It is possible to specify the maximum duration for the flux pulses with the
    `flux_pulse_max_duration` parameter.

    The function returns:
    - The full experiment pulse sequence.
    - The applied flux pulse.
    - The final `VirtualZ` pulse to be used for phase sweeping.
    """
    oscillating_natives = platform.natives.single_qubit[oscillating_qubit]
    steady_natives = platform.natives.single_qubit[steady_qubit]
    flux_channel = platform.qubits[ordered_pair[1]].flux
    assert flux_channel is not None, (
        "Flux channel must be defined for the high frequency qubit."
    )

    sequence = PulseSequence()
    # X90 on the partner qubit
    sequence += steady_natives.R(theta=np.pi / 2)
    # X
    if setup == "X":
        assert oscillating_natives.RX is not None
        sequence += oscillating_natives.RX()

    flux_pulse = _native_flux_pulse(
        platform, "iSWAP", ordered_pair, flux_pulse, flux_pulse_max_duration
    )

    sequence |= PulseSequence(
        [
            (flux_channel, Delay(duration=dt)),
            (flux_channel, flux_pulse),
            (flux_channel, Delay(duration=dt)),
        ]
    )

    vz = VirtualZ(phase=vzphase)
    sequence |= PulseSequence([(platform.qubits[oscillating_qubit].drive, vz)])

    sequence |= oscillating_natives.R(theta=np.pi / 2)

    if setup == "X":
        sequence += steady_natives.RX()

    sequence |= PulseSequence(
        [
            oscillating_natives.MZ()[0],
            steady_natives.MZ()[0],
        ]
    )

    return sequence, flux_pulse, vz


def create_sequence(
    platform: CalibrationPlatform,
    setup: Literal["I", "X"],
    oscillating_qubit: QubitId,
    steady_qubit: QubitId,
    ordered_pair: tuple[QubitId, QubitId],
    dt: float,
    flux_pulse_max_duration: float | None = None,
    flux_pulse: Pulse | None = None,
    *,
    vzphase: float = 0.0,
    native: Literal["CZ", "iSWAP"] = "CZ",
) -> tuple[PulseSequence, Pulse, VirtualZ]:
    """Create the virtual phase calibration sequence for the `native` gate.

    Dispatches to :func:`create_cz_sequence` or :func:`create_iswap_sequence`.
    """
    builders = {"CZ": _create_cz_sequence, "iSWAP": _create_iswap_sequence}
    assert native in builders, f"Unsupported native gate {native}."
    return builders[native](
        platform,
        setup,
        oscillating_qubit,
        steady_qubit,
        ordered_pair,
        dt,
        flux_pulse_max_duration,
        flux_pulse,
        vzphase=vzphase,
    )
