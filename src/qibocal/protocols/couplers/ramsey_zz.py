"""Coupler ramsey.

In this experiment, we perform a regular ZZ Ramsey protocol while applying a bias on the coupler.
This varies the cross-Kerr shift in order to find the coupler bias at which there is no net coupling.

NOTE: This assumes the qubit frequency is already well-calibrated from an existing Ramsey procedure.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
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

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitPairId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import Range, RangeLike, table_dict, table_html, to_range

__all__ = ["coupler_ramsey_zz"]


@dataclass
class CouplerRamseyResults(Results):
    """CouplerRamsey outputs."""

    nulling_point: dict[QubitPairId, float]
    """Absolute coupler nulling bias, including the acquisition DC offset."""


CouplerRamseyType = np.dtype(
    [
        ("tau", np.float64),
        ("amplitude", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
        ("prob_control", np.float64),
        ("error_control", np.float64),
    ]
)


@dataclass
class CouplerRamseyData(Data):
    """CouplerRamsey acquisition outputs."""

    data: dict[QubitPairId, npt.NDArray[CouplerRamseyType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


@dataclass
class CouplerRamseyParameters(Parameters):
    """Coupler Ramsey runcard inputs."""

    delay_range: RangeLike
    """Range of delays between RX(pi/2) pulses in ns."""
    amplitude_coupler_range: RangeLike
    """Range of coupler amplitudes."""

    @property
    def delay_sweep(self) -> Range:
        return to_range(self.delay_range)

    @property
    def amplitude_coupler_sweep(self) -> Range:
        return to_range(self.amplitude_coupler_range)


def _acquisition(
    params: CouplerRamseyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerRamseyData:

    data = CouplerRamseyData()

    flux_offsets: dict[QubitPairId, float] = {}
    coupler_pulses: list[PulseLike] = []
    delays: list[PulseLike] = []
    control_readout_pulses: dict[QubitPairId, PulseLike] = {}
    target_readout_pulses: dict[QubitPairId, PulseLike] = {}
    full_sequence = PulseSequence()

    for pair in targets:
        coupler_id = platform.calibration.two_qubits[pair].coupler
        if coupler_id is None:
            raise ValueError("Coupler ID should not be None")

        coupler_channel = platform.couplers[coupler_id].flux
        assert coupler_channel is not None, "Coupler channel should not be None"

        flux_offsets[pair] = platform.config(coupler_channel).offset

        coupler_pulse = Pulse(
            duration=params.delay_sweep[0],
            amplitude=params.amplitude_coupler_sweep[0],
            envelope=Rectangular(),
        )
        delay = Delay(duration=params.delay_sweep[0])
        coupler_pulses.append(coupler_pulse)
        delays.append(delay)

        target_qubit_id, control_qubit_id = pair

        target_qubit_natives = platform.natives.single_qubit[target_qubit_id]
        control_qubit_natives = platform.natives.single_qubit[control_qubit_id]

        target_rx90 = target_qubit_natives.R(np.pi / 2)
        target_drive_ch = platform.qubits[target_qubit_id].drive
        assert target_drive_ch is not None, "Target drive channel should not be None"

        control_rx180 = control_qubit_natives.RX()

        seq = target_rx90 + control_rx180
        preparation_duration = seq.duration

        # Start the biased Ramsey wait only after both preparation pulses finish.
        seq |= (
            PulseSequence([(coupler_channel, coupler_pulse), (target_drive_ch, delay)])
            + target_rx90
        )

        ro_ch, ro_pulse = platform.natives.single_qubit[target_qubit_id].MZ()[0]
        ro_ch_control, ro_pulse_control = platform.natives.single_qubit[
            control_qubit_id
        ].MZ()[0]

        # Keep the swept delay separate so readout follows the second RX90
        # for every wait value, even when preparation pulses differ in length.
        readout_delay = Delay(duration=preparation_duration + target_rx90.duration)

        seq += PulseSequence(
            [
                (ro_ch, readout_delay),
                (ro_ch, delay),
                (ro_ch, ro_pulse),
                (ro_ch_control, readout_delay),
                (ro_ch_control, delay),
                (ro_ch_control, ro_pulse_control),
            ]
        )

        target_readout_pulses[pair] = ro_pulse
        control_readout_pulses[pair] = ro_pulse_control

        full_sequence += seq

    # Sweep the flux-pulse duration and drive/readout delays together.
    wait_sweeper = Sweeper(
        parameter=Parameter.duration,
        range=params.delay_sweep,
        pulses=delays + coupler_pulses,
    )
    amplitude_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=params.amplitude_coupler_sweep,
        pulses=coupler_pulses,
    )

    results = platform.execute(
        sequences=[full_sequence],
        sweepers=[[amplitude_sweeper], [wait_sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # Amplitude is the outer sweep: flattening varies waits fastest.
    ampl_values = np.array(amplitude_sweeper.values)
    wait_values = np.array(wait_sweeper.values)
    for pair in targets:
        t_ro_pulse = target_readout_pulses[pair]
        c_ro_pulse = control_readout_pulses[pair]

        t_probs = np.asarray(results[t_ro_pulse.id])
        t_error = np.sqrt(t_probs * (1 - t_probs) / params.nshots)

        c_probs = np.asarray(results[c_ro_pulse.id])
        c_error = np.sqrt(c_probs * (1 - c_probs) / params.nshots)

        # Store absolute bias here; fitting and updates must not add DC again.
        data.register_qubit(
            CouplerRamseyType,
            pair,
            {
                "tau": np.tile(wait_values, len(ampl_values)),
                "amplitude": np.repeat(
                    ampl_values + flux_offsets[pair], len(wait_values)
                ),
                "prob": t_probs.ravel(),
                "error": t_error.ravel(),
                "prob_control": c_probs.ravel(),
                "error_control": c_error.ravel(),
            },
        )
    return data


def _fit(data: CouplerRamseyData) -> CouplerRamseyResults:
    nulling_point: dict[QubitPairId, float] = {}

    for pair in data.pairs:
        pair_data = data[pair]
        # Omitted pairs are treated as failed fits by downstream consumers.
        if (
            len(np.unique(pair_data["tau"])) < 2
            or not all(
                np.all(np.isfinite(pair_data[name]))
                for name in ("tau", "amplitude", "prob")
            )
            or np.any((pair_data["prob"] < 0) | (pair_data["prob"] > 1))
        ):
            log.warning(f"Coupler Ramsey fit failed for pair {pair}: invalid data.")
            continue
        amplitudes = np.unique(pair_data["amplitude"])
        variances = []

        for amp in amplitudes:
            mask = pair_data["amplitude"] == amp
            if len(np.unique(pair_data["tau"][mask])) < 2:
                break
            probs = pair_data["prob"][mask]
            variances.append(float(np.var(probs)))

        if len(variances) != len(amplitudes):
            log.warning(
                f"Coupler Ramsey fit failed for pair {pair}: insufficient delays."
            )
            continue

        # A nulling bias suppresses Ramsey fringes. Minimum variance is only
        # a heuristic: decoherence or undersampling can also flatten the trace.
        best_idx = int(np.argmin(variances))
        best_amp = float(amplitudes[best_idx])

        nulling_point[pair] = best_amp

    return CouplerRamseyResults(nulling_point)


def _build_heatmap(pair_data, field: str) -> go.Heatmap:
    """Build a 2D heatmap of ``field`` over (waits, amplitudes)."""
    amplitudes = np.unique(pair_data["amplitude"])
    waits = np.unique(pair_data["tau"])

    # Build 2D probability matrix: rows = waits, cols = amplitudes
    z = np.full((len(waits), len(amplitudes)), np.nan)
    amp_index = {amp: i for i, amp in enumerate(amplitudes)}
    tau_index = {tau: j for j, tau in enumerate(waits)}

    for row in pair_data:
        i = amp_index[row["amplitude"]]
        j = tau_index[row["tau"]]
        z[j, i] = row[field]

    return go.Heatmap(
        x=amplitudes,
        y=waits,
        z=z,
        colorscale="viridis",
        zmid=0.5,
        colorbar={"title": {"text": "Excited State Probability", "side": "right"}},
    )


def _plot(
    data: CouplerRamseyData,
    target: QubitPairId,
    fit: CouplerRamseyResults | None = None,
):
    pair_data = data[target]

    fig_target = go.Figure(_build_heatmap(pair_data, "prob"))
    fig_control = go.Figure(_build_heatmap(pair_data, "prob_control"))

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

    fitting_report = ""
    if fit is not None and target in fit.nulling_point:
        fitting_report = table_html(
            table_dict(
                str(target),
                ["Coupler nulling point [a.u.]"],
                [fit.nulling_point[target]],
            )
        )

    return [fig_target, fig_control], fitting_report


def _update(
    results: CouplerRamseyResults, platform: CalibrationPlatform, pair: QubitPairId
):
    """Update the coupler's flux offset and sweet spot with the nulling point."""
    assert platform.calibration is not None

    coupler_id = platform.calibration.two_qubits[pair].coupler
    if coupler_id is None:
        raise ValueError("Coupler ID should not be None")

    nulling_point = results.nulling_point[pair]
    update.flux_offset(nulling_point, platform, coupler_id, coupler=True)
    update.sweetspot(nulling_point, platform, coupler_id, coupler=True)


coupler_ramsey_zz = Protocol(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Ramsey ZZ Protocol for coupler nulling point calibration."""
