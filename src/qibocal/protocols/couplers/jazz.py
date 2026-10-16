"""Echo-amplified ZZ versus coupler pulse amplitude, with a phase-ramped readout."""

from dataclasses import dataclass, field

import numpy as np
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal import update
from qibocal.auto.operation import Protocol, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import RangeLike, table_dict, table_html

from .zz import (
    SETUPS,
    CouplerZZData,
    CouplerZZParameters,
    CouplerZZResults,
    CouplerZZType,
    _sweep_values,
)
from .zz import (
    _fit as fit_zz,
)
from .zz import (
    _plot as plot_zz,
)

__all__ = ["coupler_jazz"]


@dataclass
class CouplerJAZZParameters(CouplerZZParameters):
    """Coupler JAZZ inputs.

    ``delay_range`` specifies each echo half-wait in ns, not their sum.
    ``detuning`` is a readout phase-ramp rate in Hz per half-wait; it does
    not change the drive frequency. Keep |ZZ| below |detuning| to resolve sign.
    """

    delay_range: RangeLike
    """Duration of each of the two coupler flux pulses [ns]."""


@dataclass
class CouplerJAZZData(CouplerZZData):
    """Conditional JAZZ probabilities and acquisition DC offsets."""

    @property
    def frequency_scale(self) -> float:
        """The two conditional echo phases have opposite ZZ rates per half-wait."""
        return -0.5 * float(np.sign(self.detuning))


@dataclass
class CouplerJAZZResults(CouplerZZResults):
    """ZZ estimates, the minimum pulse amplitude and an absolute nulling bias."""

    nulling_point: dict[QubitPairId, float] = field(default_factory=dict)
    """Absolute minimum-|ZZ| bias, omitted for unbracketed or unresolved minima."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return pair in self.nulling_point


def _acquisition(
    params: CouplerJAZZParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerJAZZData:
    """Echo both qubits and pulse the coupler in both free-evolution windows."""
    if len(targets) != 1:
        raise ValueError("coupler_jazz requires exactly one qubit pair.")
    if (
        params.detuning is None
        or not np.isfinite(params.detuning)
        or params.detuning == 0
    ):
        raise ValueError("A finite nonzero phase-ramp detuning is required.")
    pair = targets[0]
    target, control = pair
    if target == control:
        raise ValueError("Target and control must be different qubits.")
    coupler_id = platform.calibration.two_qubits[pair].coupler
    if coupler_id is None:
        raise ValueError("Coupler ID should not be None.")
    flux = platform.couplers[coupler_id].flux
    if flux is None:
        raise ValueError("Coupler flux channel should not be None.")
    waits = _sweep_values(params.delay_range)
    amplitudes = _sweep_values(params.amplitude_range)
    if waits[0] <= 0:
        raise ValueError(
            "JAZZ half-waits and coupler pulse durations must be positive."
        )
    if len(waits) < 8:
        raise ValueError(
            "At least eight JAZZ half-waits are required for joint fitting."
        )
    if abs(params.detuning) >= 0.5e9 / np.max(np.diff(waits)):
        raise ValueError("Phase ramp must be below the wait sweep's Nyquist frequency.")
    data = CouplerJAZZData(
        detuning=params.detuning,
        flux_offsets={pair: platform.config(flux).offset},
    )
    target_natives = platform.natives.single_qubit[target]
    control_natives = platform.natives.single_qubit[control]
    drive = platform.qubits[target].drive
    control_drive = platform.qubits[control].drive
    if drive is None or control_drive is None:
        raise ValueError("Target and control drive channels are required.")

    for setup in SETUPS:
        initial = target_natives.R(theta=np.pi / 2)
        echo = target_natives.RX()
        control_echo = control_natives.RX()
        echo_duration = max(echo.duration, control_echo.duration)
        phase = 2 * np.pi * params.detuning * waits * 1e-9
        final = target_natives.R(theta=np.pi / 2, phi=phase[0])
        delay = Delay(duration=waits[0])
        first_flux = Pulse(
            duration=waits[0], amplitude=amplitudes[0], envelope=Rectangular()
        )
        second_flux = Pulse(
            duration=waits[0], amplitude=amplitudes[0], envelope=Rectangular()
        )
        sequence = initial.copy()
        sequence.append((drive, delay))
        sequence += echo
        sequence.append((drive, Delay(duration=echo_duration - echo.duration)))
        sequence.append((drive, delay))
        sequence += final
        sequence.extend(
            [
                (control_drive, Delay(duration=initial.duration)),
                (control_drive, delay),
            ]
        )
        sequence += control_echo
        sequence.extend(
            [
                (flux, Delay(duration=initial.duration)),
                (flux, first_flux),
                (flux, Delay(duration=echo_duration)),
                (flux, second_flux),
            ]
        )
        readouts = {}
        for qubit in pair:
            channel, readout = platform.natives.single_qubit[qubit].MZ()[0]
            sequence.extend(
                [
                    (
                        channel,
                        Delay(
                            duration=initial.duration + echo_duration + final.duration
                        ),
                    ),
                    (channel, delay),
                    (channel, delay),
                    (channel, readout),
                ]
            )
            readouts[qubit] = readout
        preparation = (
            control_natives.RX()
            if setup == "X"
            else PulseSequence([(control_drive, Delay(duration=control_echo.duration))])
        )
        sequence = preparation | sequence
        duration_sweeper = Sweeper(
            parameter=Parameter.duration,
            values=waits,
            pulses=[delay, first_flux, second_flux],
        )
        analysis_pulses = [pulse for _, pulse in final if isinstance(pulse, Pulse)]
        if not analysis_pulses:
            raise ValueError("The final target rotation needs a phase-sweepable pulse.")
        phase_sweeper = Sweeper(
            parameter=Parameter.relative_phase, values=phase, pulses=analysis_pulses
        )
        amplitude_sweeper = Sweeper(
            parameter=Parameter.amplitude,
            values=amplitudes,
            pulses=[first_flux, second_flux],
        )
        acquired = platform.execute(
            [sequence],
            [[amplitude_sweeper], [duration_sweeper, phase_sweeper]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
        data.register_qubit(
            CouplerZZType,
            (pair, setup),
            {
                "wait": np.tile(waits, len(amplitudes)),
                "amplitude": np.repeat(amplitudes, len(waits)),
                "prob": np.asarray(acquired[readouts[target].id]).ravel(),
            },
        )
    return data


def _fit(data: CouplerJAZZData) -> CouplerJAZZResults:
    """Convert the two echoed fringe frequencies to ZZ and bracket its minimum."""
    fitted = fit_zz(data)
    results = CouplerJAZZResults(**fitted.params)
    for pair, point in results.zz_free_amplitude.items():
        interval = results.fit_amplitudes[pair]
        if not interval or not interval[0] < point < interval[1]:
            log.warning(
                f"JAZZ nulling bias for pair {pair} is not bracketed; no update."
            )
            continue
        if pair not in data.flux_offsets or not np.isfinite(data.flux_offsets[pair]):
            log.warning(
                f"JAZZ nulling bias for pair {pair} lacks a finite acquisition DC offset."
            )
            continue
        results.nulling_point[pair] = data.flux_offsets[pair] + point
    return results


def _plot(
    data: CouplerJAZZData,
    target: QubitPairId,
    fit: CouplerJAZZResults | None = None,
):
    """Report both control preparations, ZZ uncertainties and the fitted minimum."""
    figures, report = plot_zz(data, target, fit)
    figures[0].update_xaxes(title_text="Echo half-wait [ns]")
    if fit is not None and target in fit.nulling_point:
        report += table_html(
            table_dict(
                str(target),
                ["Coupler nulling bias (minimum |ZZ|) [a.u.]"],
                [fit.nulling_point[target]],
            )
        )
    return figures, report


def _update(
    results: CouplerJAZZResults, platform: CalibrationPlatform, target: QubitPairId
):
    """Apply an absolute bias; repeated updates never accumulate the DC offset."""
    coupler_id = platform.calibration.two_qubits[target].coupler
    if coupler_id is None:
        raise ValueError("Coupler ID should not be None.")
    point = results.nulling_point[target]
    update.flux_offset(point, platform, coupler_id, coupler=True)
    update.sweetspot(point, platform, coupler_id, coupler=True)


coupler_jazz = Protocol(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Coupler JAZZ nulling-point calibration."""
