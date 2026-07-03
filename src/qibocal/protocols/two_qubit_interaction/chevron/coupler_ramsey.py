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
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, Protocol, QubitPairId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import table_dict, table_html


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

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""
    amplitude_coupler_min: float
    """Amplitude coupler minimum."""
    amplitude_coupler_max: float
    """Amplitude coupler maximum."""
    amplitude_coupler_step: float
    """Amplitude coupler step."""


def _acquisition(
    params: CouplerRamseyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
):
    waits = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )
    amplitudes = np.arange(
        params.amplitude_coupler_min,
        params.amplitude_coupler_max,
        params.amplitude_coupler_step,
    )

    sequence = PulseSequence()

    coupler_pulse = Pulse(duration=0, amplitude=0, envelope=Rectangular())
    delay = Delay(duration=0)
    coupler_amplitude_sweeper = Sweeper(
        parameter=Parameter.amplitude, values=amplitudes, pulses=[coupler_pulse]
    )
    duration_sweeper = Sweeper(
        parameter=Parameter.duration, values=waits, pulses=[delay, coupler_pulse]
    )

    for pair in targets:
        coupler_channel = platform.couplers[
            platform.calibration.two_qubits[pair].coupler
        ].flux
        target_qubit_id, control_qubit_id = pair
        target_qubit = platform.qubits[target_qubit_id]
        rx90 = platform.natives.single_qubit[target_qubit_id].R(np.pi / 2)

        seq = PulseSequence()
        seq += rx90
        seq |= [
            (coupler_channel, coupler_pulse),
            (target_qubit.acquisition, delay),
            (target_qubit.drive, delay),
        ]
        seq += (
            rx90 | platform.natives.single_qubit[target_qubit_id].MZ()
        ).align_to_delays()

        seq += platform.natives.single_qubit[control_qubit_id].RX()
        # Add the partial sequence to the full sequence
        sequence += seq

    results = platform.execute(
        sequences=[sequence],
        sweepers=[[duration_sweeper], [coupler_amplitude_sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    data = CouplerRamseyData()

    for pair in targets:
        target_qubit_id, control_qubit_id = pair
        ro_pulse = list(sequence.channel(platform.qubits[target_qubit_id].acquisition))[
            -1
        ]
        prob_2d = results[ro_pulse.id]

        for wait, prob_1d in zip(waits, prob_2d):
            for amplitude, prob in zip(amplitudes, prob_1d):
                error = np.sqrt(prob * (1 - prob) / params.nshots)
                data.register_qubit(
                    CouplerRamseyType,
                    pair,
                    {
                        "tau": np.array([wait]),
                        "amplitude": np.array([amplitude]),
                        "prob": np.array([prob]),
                        "error": np.array([error]),
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


def _plot(
    data: CouplerRamseyData,
    target: QubitPairId,
    fit: CouplerRamseyResults | None = None,
):
    pair_data = data[target]
    amplitudes = np.unique(pair_data["amplitude"])
    waits = np.unique(pair_data["tau"])

    # Build 2D probability matrix: rows = waits, cols = amplitudes
    z = np.full((len(waits), len(amplitudes)), np.nan)
    amp_index = {amp: i for i, amp in enumerate(amplitudes)}
    tau_index = {tau: j for j, tau in enumerate(waits)}

    for row in pair_data:
        i = amp_index[row["amplitude"]]
        j = tau_index[row["tau"]]
        z[j, i] = row["prob"]

    fig = go.Figure(
        go.Heatmap(
            x=amplitudes,
            y=waits,
            z=z,
            colorscale="viridis",
            zmid=0.5,
            colorbar=dict(title={"text": "Excited State Probability", "side": "right"}),
        )
    )

    if fit is not None and target in fit.offset_adjustment:
        best_amp = fit.offset_adjustment[target][0]
        fig.add_vline(
            x=best_amp,
            line=dict(color="black", dash="dash", width=2),
            annotation_text=f"Offset adjustment: {best_amp:.4f}",
            annotation_position="right",
        )

    fig.update_layout(
        yaxis_title="Time [ns]",
        xaxis_title="Amplitude [a.u.]",
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

    return [fig], fitting_report


def _update(
    results: CouplerRamseyResults, platform: CalibrationPlatform, pair: QubitPairId
):
    pass


coupler_ramsey = Protocol(_acquisition, _fit, _plot, _update)
"""Coupler ramsey zz protocol."""
