"""JAZZ (Joint Amplification of ZZ) versus coupler flux-pulse amplitude.

A single echoed Ramsey sequence on the target qubit, with a simultaneous
:math:`\\pi` pulse on both qubits at the echo::

    target:  RX90 -- w -- RX  -- w -- R90(phi) -- MZ
    control:      -- w -- RX  --
    coupler:      -- flux --     -- flux --

The control qubit spends the first half-wait in :math:`|0\\rangle` and the second
in :math:`|1\\rangle`, while the target is flipped. Any unconditional target
detuning (drive miscalibration, coupler-induced shifts) is echoed away, whereas
the conditional ZZ phases of the two halves add up: the target accumulates
:math:`2\\pi\\zeta w` per half-wait :math:`w`. No second control preparation is
needed, unlike conditional Ramsey (:mod:`.zz`).

A software phase ramp :math:`\\phi = 2\\pi\\delta w` on the final rotation
displaces the fringe to :math:`f = |\\delta + \\zeta|`, so the sign of
:math:`\\zeta = \\mathrm{sign}(\\delta) f - \\delta` is resolved provided
:math:`|\\zeta| < |\\delta|`.

Reference:
    - Ku et al., "Suppression of Unwanted ZZ Interactions in a Hybrid
      Two-Qubit System", PRL 125, 200504 (2020).
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.constants as const
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseLike,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Protocol, QubitPairId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    Range,
    RangeLike,
    quinn_fernandes_algorithm,
    table_dict,
    table_html,
    to_range,
)
from qibocal.protocols.zz_interaction.jazz import (
    jazz_fit,
    jazz_sequence,
)
from qibocal.protocols.zz_interaction.utils import (
    ZZInteractionData,
    ZZInteractionParameters,
)

from .ramsey_zz import _build_heatmap

__all__ = ["coupler_jazz"]


@dataclass
class CouplerJAZZParameters(ZZInteractionParameters):
    """Coupler JAZZ inputs.

    ``delay_range`` specifies each echo half-wait in ns, not their sum.
    ``detuning`` is a readout phase-ramp rate in Hz per half-wait; it does
    not change the drive frequency. Keep |ZZ| below |detuning| to resolve sign.
    """

    coupler_amp_range: RangeLike
    """Amplitude of the coupler flux pulse."""

    @property
    def _coupler_range(self) -> Range:
        return to_range(self.coupler_amp_range)


CouplerZZType = np.dtype(
    [
        ("delay", np.float64),
        ("amplitude", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
    ]
)


@dataclass
class CouplerJAZZData(ZZInteractionData):
    """JAZZ target probabilities and acquisition DC offsets."""

    """Phase-ramp rate of the final target rotation [Hz per half-wait]."""
    couplers_offsets: dict[QubitPairId, float] = field(default_factory=dict)
    """Acquisition DC coupler offsets, for absolute-bias calibration."""
    data: dict[QubitPairId, npt.NDArray[CouplerZZType]] = field(default_factory=dict)
    """Raw data acquired."""

    def amplitudes(self, pair: QubitPairId) -> npt.NDArray:
        return np.unique(self.data[pair]["amplitude"])

    def return_data_per_amplitude(
        self, pair: QubitPairId, amp: float
    ) -> npt.NDArray[CouplerZZType]:
        return self.data[pair][np.array(self.data[pair])["amplitude"] == amp]


@dataclass
class CouplerJAZZResults(Results):
    """ZZ estimates, the minimum pulse amplitude and an absolute nulling bias."""

    zz: dict[QubitPairId, list[float]] = field(default_factory=dict)
    nulling_point: dict[QubitPairId, float] = field(default_factory=dict)
    """Absolute minimum-|ZZ| bias, omitted for unbracketed or unresolved minima."""
    fitting_params: dict[QubitPairId, dict[float, list[float]]] = field(
        default_factory=dict
    )

    def __contains__(self, pair: QubitPairId) -> bool:
        return pair in self.nulling_point


def _acquisition(
    params: CouplerJAZZParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerJAZZData:
    """Echo both qubits and pulse the coupler in both free-evolution windows."""

    complete_sequence = PulseSequence()

    delay = Delay(duration=params.delay_range[0])
    coupler_pulse = Pulse(
        duration=params.delay_range[0],
        amplitude=params._coupler_range[0],
        envelope=Rectangular(),
    )

    coupler_offsets: dict[QubitPairId, float] = {}
    target_ro_pulses: dict[QubitPairId, PulseLike] = {}
    spectator_ro_pulses: dict[QubitPairId, PulseLike] = {}
    for pair in targets:
        target, spectator = pair

        # target qubit readout
        target_natives = platform.natives.single_qubit[target]
        target_ro_channel, target_ro_pulse = target_natives.MZ()[0]

        # spectator qubit readout
        spectator_natives = platform.natives.single_qubit[spectator]
        spectator_ro_channel, spectator_ro_pulse = spectator_natives.MZ()[0]

        acquisition_seq = PulseSequence(
            [
                (target_ro_channel, target_ro_pulse),
                (spectator_ro_channel, spectator_ro_pulse),
            ]
        )
        target_ro_pulses[pair] = target_ro_pulse
        spectator_ro_pulses[pair] = spectator_ro_pulse

        delay_sequence, flip_sequence = jazz_sequence(
            calibration_platform=platform,
            target=target,
            spectator=spectator,
            delay=delay,
        )

        coupler_id = platform.calibration.two_qubits[pair].coupler
        if coupler_id is None:
            raise ValueError("Coupler ID should not be None")

        coupler_channel = platform.couplers[coupler_id].flux
        assert coupler_channel is not None, "Coupler channel should not be None"

        coupler_offsets[pair] = platform.config(coupler_channel).offset

        # adding coupler flux pulse to the delay sequence
        coupler_flux_seq = PulseSequence([(coupler_channel, coupler_pulse)])
        delay_sequence += coupler_flux_seq

        # adding coupler idle to the flip sequence
        coupler_idle_seq = PulseSequence(
            [(coupler_channel, Delay(duration=flip_sequence.duration))]
        )
        flip_sequence += coupler_idle_seq

        target_rx90_seq = target_natives.R(theta=np.pi / 2)
        target_ry90_seq = target_natives.R(theta=np.pi / 2, phi=np.pi / 2)

        complete_sequence += (
            target_rx90_seq  # rotating target on the equator
            | delay_sequence  # first delay sequence
            | flip_sequence  # spectator and target flipping
            | delay_sequence  # second delay sequence
            | target_ry90_seq  # rotating target on the equator with a pi/2 phase shift
            | acquisition_seq  # measure both qubits
        )

    data = CouplerJAZZData(
        couplers_offsets=coupler_offsets,
    )

    duration_sweeper = Sweeper(
        parameter=Parameter.duration,
        range=params.delay_range,
        pulses=[delay, coupler_pulse],
    )

    amplitude_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=params._coupler_range,
        pulses=[coupler_pulse],
    )

    acquired = platform.execute(
        [complete_sequence],
        [[amplitude_sweeper], [duration_sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # Amplitude is the outer sweep: flattening varies delays fastest.
    ampl_values = np.array(amplitude_sweeper.values)
    delay_values = np.array(duration_sweeper.values)
    for pair in targets:
        t_ro_pulse = target_ro_pulses[pair]
        s_ro_pulse = spectator_ro_pulses[pair]

        t_probs = np.asarray(acquired[t_ro_pulse.id])
        t_error = np.sqrt(t_probs * (1 - t_probs) / params.nshots)

        s_probs = np.asarray(acquired[s_ro_pulse.id])
        s_error = np.sqrt(s_probs * (1 - s_probs) / params.nshots)

        # Store absolute bias here; fitting and updates must not add DC again.
        data.register_qubit(
            CouplerZZType,
            pair,
            {
                "delay": np.tile(delay_values, len(ampl_values)),
                "amplitude": np.repeat(
                    ampl_values + coupler_offsets[pair], len(delay_values)
                ),
                "prob": t_probs.ravel(),
                "error": t_error.ravel(),
                "prob_control": s_probs.ravel(),
                "error_control": s_error.ravel(),
            },
        )

    return data


def _fit(data: CouplerJAZZData) -> CouplerJAZZResults:
    """Convert the echoed fringe frequency to ZZ and bracket its minimum."""

    zz_strenghts: dict[QubitPairId, list[float]] = defaultdict(list)
    fitting_params: dict[QubitPairId, dict[float, list[float]]] = defaultdict(dict)
    for pair in data.pairs:
        amplitudes = data.amplitudes(pair)

        for amp in amplitudes:
            amp_data = data.return_data_per_amplitude(pair, amp)

            try:
                # popt, zz_estimated = _jazz_fitting_process(
                #     probs=amp_data['prob'],
                #     delays=amp_data['delay'],
                #     err=amp_data['error'],
                # )

                probs = amp_data["prob"]
                delays = amp_data["delay"]
                err = amp_data["error"]

                probs_max = np.max(probs)
                probs_min = np.min(probs)
                d_max = np.max(delays)
                d_min = np.min(delays)
                delta_probs = probs_max - probs_min
                delta_delay = d_max - d_min
                min_max_probs = (probs - probs_min) / delta_probs
                min_max_delays = (delays - d_min) / delta_delay
                err = err / delta_probs

                omega = quinn_fernandes_algorithm(
                    min_max_probs, min_max_delays, speedup_flag=True
                ).tolist()

                # zz_strenghts[pair].append(float(zz_estimated[0]))
                zz_strenghts[pair].append(float(omega))
                fitting_params[pair][amp] = [0] * 4

            # here we are catching two possible errors: fitting error in curve_fit
            # call or error in estimating coupling strenght because anharmonicities
            # are unknown
            except (RuntimeError, ValueError) as e:
                log.warning(f"JAZZ fit for pair {pair}: {e}.")

    return CouplerJAZZResults(
        zz=zz_strenghts,
        nulling_point={
            k: data.amplitudes(k)[np.argmin(np.abs(v))] for k, v in zz_strenghts.items()
        },
        fitting_params=fitting_params,
    )


def _plot(
    data: CouplerJAZZData,
    target: QubitPairId,
    fit: CouplerJAZZResults | None = None,
):
    """Report the echoed fringes, ZZ uncertainties and the fitted minimum."""
    pair_data = data[target]

    fig_target = go.Figure(_build_heatmap(pair_data, "prob"))
    fig_control = go.Figure(_build_heatmap(pair_data, "prob_control"))

    figures: list[go.Figure] = [fig_target, fig_control]

    if fit is not None and target in fit.nulling_point:
        best_amp = fit.nulling_point[target]
        fig_target.add_vline(
            x=best_amp,
            line={"color": "black", "dash": "dash", "width": 2},
        )

    fig_target.update_layout(
        yaxis_title="Time [ns]",
        xaxis_title="Coupler Offset [a.u.]",
        title="Target qubit",
    )
    fig_control.update_layout(
        yaxis_title="Time [ns]",
        xaxis_title="Coupler Offset [a.u.]",
        title="Control qubit",
    )

    figures_1d: list[go.Figure] = []
    for amp in data.amplitudes(target):
        if (
            fit is not None
            and target in fit.fitting_params
            and amp in fit.fitting_params[target]
        ):
            amp_data = data.return_data_per_amplitude(target, amp)
            times = amp_data["delay"]
            probs = amp_data["prob"]

            t_linspace = np.linspace(times[0], times[-1], 200)
            best_fit = fit.fitting_params[target][amp]

            fig_1d = go.Figure()
            fig_1d.add_scatter(
                x=times, y=probs, mode="markers", name=f"Data (amp={np.round(amp, 3)})"
            )
            fig_1d.add_scatter(
                x=t_linspace,
                y=jazz_fit(t_linspace, *best_fit),
                mode="lines",
                name="Fit",
            )
            figures_1d.append(fig_1d)

    fitting_report = ""
    if fit is not None and target in fit.nulling_point:
        fitting_report = table_html(
            table_dict(
                str(target),
                ["Coupler nulling point [a.u.]"],
                [fit.nulling_point[target]],
            )
        )

    if fit is not None:
        zz_fig = go.Figure(
            go.Scatter(
                x=data.amplitudes(target),
                y=np.array(fit.zz[target]) * const.milli,
                mode="lines+markers",
                name="ZZ",
            )
        )
        zz_fig.update_layout(
            xaxis_title="Coupler pulse amplitude [a.u.]",
            yaxis_title="ZZ coupling [kHz]",
            showlegend=True,
        )
        figures.append(zz_fig)

    return figures + figures_1d, fitting_report


def _update(
    results: CouplerJAZZResults, platform: CalibrationPlatform, target: QubitPairId
):
    """Apply an absolute bias; repeated updates never accumulate the DC offset."""
    # assert platform.calibration is not None
    # coupler_id = platform.calibration.two_qubits[target].coupler
    # if coupler_id is None:
    #     raise ValueError("Coupler ID should not be None.")
    # point = results.nulling_point[target]
    # update.flux_offset(point, platform, coupler_id, coupler=True)
    # update.sweetspot(point, platform, coupler_id, coupler=True)


coupler_jazz = Protocol(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Coupler JAZZ nulling-point calibration."""
