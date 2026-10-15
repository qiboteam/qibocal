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
from qibocal.protocols.utils import Range, RangeLike, table_dict, table_html, to_range


@dataclass
class CouplerRamseyResults(Results):
    """CouplerRamsey outputs."""

    offset_adjustment: dict[QubitPairId, float]
    """Coupler offset adjustment."""


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
    """T2Signal runcard inputs."""

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
):

    data = CouplerRamseyData()

    coupler_pulse = Pulse(
        duration=params.delay_sweep[0],
        amplitude=params.amplitude_coupler_sweep[0],
        envelope=Rectangular(),
    )
    delay = Delay(duration=params.delay_sweep[0])

    readout_pulses: dict[QubitPairId, tuple[PulseLike, PulseLike]] = {}
    full_sequence = PulseSequence()
    for pair in targets:
        # Get the coupler ID for the current qubit pair
        coupler_id = platform.calibration.two_qubits[pair].coupler
        # Raise an error if the coupler ID is not found
        if coupler_id is None:
            raise ValueError("Coupler ID should not be None")

        # Get the flux channel for the current coupler ID
        coupler_channel = platform.couplers[coupler_id].flux
        assert coupler_channel is not None, "Coupler channel should not be None"

        target_qubit_id, control_qubit_id = pair

        # Get the target qubit ntive gates
        target_qubit_natives = platform.natives.single_qubit[target_qubit_id]
        # Get the control qubit native gates
        control_qubit_natives = platform.natives.single_qubit[control_qubit_id]

        target_rx90 = target_qubit_natives.R(np.pi / 2)
        target_drive_ch = platform.qubits[target_qubit_id].drive
        assert target_drive_ch is not None, "Target drive channel should not be None"

        control_rx180 = control_qubit_natives.RX()

        # initialize the sequence with the first set of pulses
        seq = target_rx90 + control_rx180

        seq |= (
            PulseSequence([(coupler_channel, coupler_pulse), (target_drive_ch, delay)])
            + target_rx90
        )

        # Create the readout sequence for the target and control qubits
        ro_ch, ro_pulse = platform.natives.single_qubit[target_qubit_id].MZ()[0]
        ro_ch_control, ro_pulse_control = platform.natives.single_qubit[
            control_qubit_id
        ].MZ()[0]
        # define the readout sequence
        ro_seq = PulseSequence(
            [
                (ro_ch, delay),
                (ro_ch, Delay(duration=2 * target_rx90.duration)),
                (ro_ch, ro_pulse),
                (ro_ch_control, delay),
                (ro_ch_control, Delay(duration=2 * target_rx90.duration)),
                (ro_ch_control, ro_pulse_control),
            ]
        )

        # save the readout pulses for the current qubit pair
        readout_pulses[pair] = (ro_pulse, ro_pulse_control)

        # Add the partial sequence to the full sequence
        full_sequence += seq + ro_seq

    wait_sweeper = Sweeper(
        parameter=Parameter.duration,
        range=params.delay_sweep,
        pulses=[delay, coupler_pulse],
    )
    amplitude_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=params.amplitude_coupler_sweep,
        pulses=[coupler_pulse],
    )

    results = platform.execute(
        sequences=[full_sequence],
        sweepers=[[amplitude_sweeper], [wait_sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    ampl_values = np.array(amplitude_sweeper.values)
    wait_values = np.array(wait_sweeper.values)
    for pair in targets:
        ro_pulse, ro_pulse_control = readout_pulses[pair]
        # probs has shape (len(amplitudes), len(waits))
        probs = np.asarray(results[ro_pulse.id])
        error = np.sqrt(probs * (1 - probs) / params.nshots)
        probs_control = np.asarray(results[ro_pulse_control.id])
        error_control = np.sqrt(probs_control * (1 - probs_control) / params.nshots)
        data.register_qubit(
            CouplerRamseyType,
            pair,
            {
                "tau": np.tile(wait_values, len(ampl_values)),
                "amplitude": np.repeat(ampl_values, len(wait_values)),
                "prob": probs.ravel(),
                "error": error.ravel(),
                "prob_control": probs_control.ravel(),
                "error_control": error_control.ravel(),
            },
        )
    return data


def _fit(data: CouplerRamseyData) -> CouplerRamseyResults:
    offset_adjustment: dict[QubitPairId, list[float]] = {}

    for pair in data.pairs:
        pair_data = data[pair]
        amplitudes = np.unique(pair_data["amplitude"])
        variances = []

        for amp in amplitudes:
            mask = pair_data["amplitude"] == amp
            probs = pair_data["prob"][mask]
            variances.append(float(np.var(probs)))

        best_idx = int(np.argmin(variances))
        best_amp = float(amplitudes[best_idx])

        offset_adjustment[pair] = [best_amp, 0.0]

    return CouplerRamseyResults(offset_adjustment=offset_adjustment)


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

    if fit is not None and target in fit.offset_adjustment:
        best_amp = fit.offset_adjustment[target][0]
        fig_target.add_vline(
            x=best_amp,
            line={"color": "black", "dash": "dash", "width": 2},
            annotation_text=f"Offset adjustment: {best_amp:.4f}",
            annotation_position="right",
        )

    fig_target.update_layout(
        yaxis_title="Time [ns]",
        xaxis_title="Amplitude [a.u.]",
        title="Target qubit",
    )
    fig_control.update_layout(
        yaxis_title="Time [ns]",
        xaxis_title="Amplitude [a.u.]",
        title="Control qubit",
    )

    fitting_report = ""
    if fit is not None and target in fit.offset_adjustment:
        fitting_report = table_html(
            table_dict(
                str(target),
                ["Offset adjustment [a.u.]"],
                [fit.offset_adjustment[target]],
                display_error=True,
            )
        )

    return [fig_target, fig_control], fitting_report


def _update(
    results: CouplerRamseyResults, platform: CalibrationPlatform, pair: QubitPairId
):
    """Update the coupler's flux offset and sweet spot from Ramsey results."""
    assert platform.calibration is not None

    coupler_id = platform.calibration.two_qubits[pair].coupler
    if coupler_id is None:
        raise ValueError("Coupler ID should not be None")

    nulling_point = results.offset_adjustment[pair]
    update.flux_offset(nulling_point, platform, coupler_id)
    update.sweetspot(nulling_point, platform, coupler_id)


coupler_ramsey = Protocol(_acquisition, _fit, _plot, _update)
"""Ramsey Protocol object."""
