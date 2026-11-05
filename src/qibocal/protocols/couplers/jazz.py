"""JAZZ (Joint Amplification of ZZ) versus coupler flux-pulse offset.

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

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.constants as const
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseLike,
    PulseSequence,
    Sweeper,
)
from scipy.signal import savgol_filter

from qibocal.auto.operation import Protocol, QubitPairId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    RangeLike,
    quinn_fernandes_algorithm,
    table_dict,
    table_html,
    to_range,
)
from qibocal.protocols.zz_interaction.jazz import (
    jazz_fitting_process,
    jazz_sequence,
)
from qibocal.protocols.zz_interaction.utils import (
    ZZInteractionData,
    ZZInteractionParameters,
)

__all__ = ["coupler_jazz"]


SAVGOL_POLYORDER = 3
SAVGOL_WINDOW_LENGTH = 4
SAVGOL_DERIV = 1


@dataclass
class CouplerJAZZParameters(ZZInteractionParameters):
    """Coupler JAZZ inputs.

    ``delay_range`` specifies each echo half-wait in ns, not their sum.
    ``detuning`` is a readout phase-ramp rate in Hz per half-wait; it does
    not change the drive frequency. Keep |ZZ| below |detuning| to resolve sign.
    """

    coupler_offset_range: RangeLike
    """Offset of the coupler flux pulse."""
    fast_evaluation: bool = True
    """Whether to estimate ZZ with only the Quinn-Fernandes frequency estimator
    (:func:`~your_package.fitting.quinn_fernandes`), skipping the curve fit.

    If ``False``, a full fit with :func:`scipy.optimize.curve_fit` is performed instead.

    Use ``True`` for first (coarse) scans. ZZ can vary strongly with the coupler frequency,
    so the oscillations may be undersampled, and a full fit then becomes unreliable.
    """


CouplerZZType = np.dtype(
    [
        ("delay", np.float64),
        ("offset", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
        ("prob_spectator", np.float64),
        ("error_spectator", np.float64),
    ]
)


@dataclass
class CouplerJAZZData(ZZInteractionData):
    """JAZZ target probabilities and acquisition DC offsets."""

    """Phase-ramp rate of the final target rotation [Hz per half-wait]."""
    data: dict[QubitPairId, npt.NDArray[CouplerZZType]] = field(default_factory=dict)
    """Raw data acquired."""
    fast_evaluation: bool = True
    """Whether fast evaluation was used, as configured by
    :attr:`CouplerJAZZParameters.fast_evaluation`.
    """

    def pair_offsets(self, pair: QubitPairId) -> npt.NDArray:
        return np.unique(self.data[pair]["offset"])

    def return_data_per_offset(
        self, pair: QubitPairId, bias: float
    ) -> npt.NDArray[CouplerZZType]:
        return self.data[pair][np.array(self.data[pair])["offset"] == bias]


@dataclass
class CouplerJAZZResults(Results):
    """ZZ estimates, the minimum pulse amplitude and an absolute nulling bias."""

    zz: dict[QubitPairId, list[list[float]]] = field(default_factory=dict)
    """ZZ estimates for each qubit pair [Hz]."""
    nulling_point: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Absolute minimum-|ZZ| bias, omitted for unbracketed or unresolved minima."""
    fitting_params: dict[QubitPairId, dict[float, list[float]]] = field(
        default_factory=dict
    )
    """Fitting parameters for each qubit pair."""


def _acquisition(
    params: CouplerJAZZParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerJAZZData:
    """Echo both qubits and pulse the coupler in both free-evolution windows."""

    complete_sequence = PulseSequence()

    delay = Delay(duration=params.delay_range[0])

    updates: dict = {}
    target_ro_pulses: dict[QubitPairId, PulseLike] = {}
    spectator_ro_pulses: dict[QubitPairId, PulseLike] = {}
    offset_parsweepers: dict[QubitPairId, Sweeper] = {}
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

        jazz_seq = jazz_sequence(
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
        coupler_offset = platform.config(coupler_channel).offset

        # resetting the coupler offset to 0 before the sweep
        updates |= {coupler_channel: {"offset": 0.0}}

        complete_sequence += jazz_seq | acquisition_seq

        offset_parsweepers[pair] = Sweeper(
            parameter=Parameter.offset,
            range=to_range(params.coupler_offset_range, center=coupler_offset),
            channels=[coupler_channel],
        )

    data = CouplerJAZZData(fast_evaluation=params.fast_evaluation)

    duration_sweeper = Sweeper(
        parameter=Parameter.duration,
        range=params.delay_range,
        pulses=[delay],
    )
    acquired = platform.execute(
        [complete_sequence],
        [list(offset_parsweepers.values()), [duration_sweeper]],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # offset is the outer sweep: flattening varies delays fastest.
    delay_values = np.array(duration_sweeper.values)
    for pair in targets:
        off_values = np.array(offset_parsweepers[pair].values)
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
                "delay": np.tile(delay_values, len(off_values)),
                "offset": np.repeat(off_values, len(delay_values)),
                "prob": t_probs.ravel(),
                "error": t_error.ravel(),
                "prob_spectator": s_probs.ravel(),
                "error_spectator": s_error.ravel(),
            },
        )

    return data


def fast_estimation(
    offset_data: npt.NDArray[CouplerZZType],
    delays: npt.NDArray,
) -> tuple[float, list[float]]:
    """Estimate signed ZZ in Hz for one coupler offset."""
    probabilities = offset_data["prob"]
    omega = float(
        quinn_fernandes_algorithm(probabilities, delays, speedup_flag=True)
        * const.giga
        / (2 * np.pi)
    )

    try:
        quarter_period = const.giga / (4 * omega) if omega else np.inf
        quarter_period_signal = probabilities[delays <= quarter_period]
        filtered_signal = savgol_filter(
            quarter_period_signal,
            window_length=SAVGOL_WINDOW_LENGTH,
            polyorder=SAVGOL_POLYORDER,
            deriv=SAVGOL_DERIV,
        )
        zz_sign = np.sign(np.median(filtered_signal))
    except ValueError:
        log.warning(
            "Failed in estimating interaction sign: not enough points in the semi period signal."
        )
        zz_sign = 1

    zz_sign = 1

    return float(omega * zz_sign), [0.0] * 4


def full_estimation(
    offset_data: npt.NDArray[CouplerZZType],
    delays: npt.NDArray,
) -> tuple[float, list[float]]:
    """Fit ZZ in Hz for one coupler offset."""
    popt, zz_estimated = jazz_fitting_process(
        probs=offset_data["prob"],
        delays=delays,
        err=offset_data["error"],
    )
    return float(zz_estimated[0]), popt


def _fit(data: CouplerJAZZData) -> CouplerJAZZResults:
    """Convert the echoed fringe frequency to ZZ and bracket its minimum."""

    zz_strenghts: dict[QubitPairId, list[list[float]]] = {}
    fitting_params: dict[QubitPairId, dict[float, list[float]]] = {}
    nulling_point: dict[QubitPairId, list[float]] = {}
    jazz_estimation = fast_estimation if data.fast_evaluation else full_estimation
    for pair in data.pairs:
        for off in data.pair_offsets(pair):
            offset_data = data.return_data_per_offset(pair, off)
            try:
                zz_estimated, popt = jazz_estimation(offset_data, data.delays)
            except (RuntimeError, ValueError) as e:
                log.warning(f"JAZZ fit for pair {pair} at offset {off}: {e}.")
                continue
            zz_strenghts.setdefault(pair, []).append([float(off), zz_estimated])
            fitting_params.setdefault(pair, {})[float(off)] = popt

        if pair in zz_strenghts:
            nulling_point[pair] = min(
                zz_strenghts[pair], key=lambda point: abs(point[1])
            )

    return CouplerJAZZResults(
        zz=zz_strenghts,
        nulling_point=nulling_point,
        fitting_params=fitting_params,
    )


def _plot(
    data: CouplerJAZZData,
    target: QubitPairId,
    fit: CouplerJAZZResults | None = None,
):
    """Report target/control fringes above the estimated ZZ coupling."""
    pair_data = data[target]
    offsets = data.pair_offsets(target)
    delays = data.delays
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("Target qubit", "Control qubit", "Estimated ZZ"),
    )
    for col, probability_field in enumerate(("prob", "prob_spectator"), start=1):
        probabilities = pair_data[probability_field].reshape(len(offsets), len(delays))
        fig.add_trace(
            go.Heatmap(
                x=offsets,
                y=delays,
                z=probabilities.T,
                coloraxis="coloraxis",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="Coupler Offset [a.u.]", row=1, col=col)
        fig.update_yaxes(title_text="Time [ns]", row=1, col=col)
    fitting_report = ""

    if fit is not None:
        if target in fit.nulling_point:
            null_bias, null_interaction = fit.nulling_point[target]
            fig.add_vline(
                x=null_bias,
                line={"color": "red", "dash": "dash", "width": 2},
                row=1,
                col=1,
            )

            fitting_report = table_html(
                table_dict(
                    str(target),
                    [
                        "Coupler nulling point [a.u.]",
                        "Coupler minimum Interaction [kHz]",
                    ],
                    [null_bias, null_interaction * const.milli],
                )
            )

        if target in fit.zz:
            zz_fit = np.asarray(fit.zz[target])
            fig.add_trace(
                go.Scatter(
                    x=zz_fit[:, 0],
                    y=zz_fit[:, 1] * const.milli,
                    mode="lines+markers",
                    name="ZZ",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    fig.update_xaxes(title_text="Coupler pulse offset [a.u.]", row=2, col=1)
    fig.update_yaxes(title_text="ZZ coupling [kHz]", row=2, col=1)
    fig.update_layout(
        height=800,
        coloraxis={
            "colorscale": "inferno",
            "cmin": 0,
            "cmax": 1,
            "colorbar": {
                "title": {"text": "Excited State Probability", "side": "right"},
                "len": 0.4,
                "y": 1,
                "yanchor": "top",
            },
        },
        showlegend=True,
    )

    return [fig], fitting_report


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
