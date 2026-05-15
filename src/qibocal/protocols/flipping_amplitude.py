"""Flipping experiment sweeping number of flips and pulse amplitude."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, PulseSequence, Readout

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from .flipping import flipping_sequence

__all__ = ["flipping_amplitude"]


@dataclass
class FlippingAmplitudeParameters(Parameters):
    """FlippingAmplitude runcard inputs."""

    nflips_max: int = 21
    """Maximum number of flips ([RX(pi) - RX(pi)] sequences)."""
    nflips_step: int = 1
    """Step size for the number of consecutive flips."""
    delta_amplitude_min: float = -0.05
    """Minimum amplitude delta relative to the native pulse amplitude."""
    delta_amplitude_max: float = 0.05
    """Maximum amplitude delta relative to the native pulse amplitude."""
    delta_amplitude_step: float = 0.001
    """Amplitude delta step."""
    rx90: bool = False
    """Calibration of native pi pulse, if true calibrates pi/2 pulse."""

    def __post_init__(self):
        if not isinstance(self.nflips_max, int):
            raise TypeError(
                f"nflips_max must be int, got {type(self.nflips_max).__name__}"
            )
        if not isinstance(self.nflips_step, int):
            raise TypeError(
                f"nflips_step must be int, got {type(self.nflips_step).__name__}"
            )
        if not isinstance(self.rx90, bool):
            raise TypeError(f"rx90 must be boolean, got {type(self.rx90).__name__}")
        if self.nflips_max <= 0:
            raise ValueError("nflips_max must be greater than 0.")
        if self.nflips_step <= 0:
            raise ValueError("nflips_step must be greater than 0.")


@dataclass
class FlippingAmplitudeResults(Results):
    """FlippingAmplitude outputs."""

    amplitude: dict[QubitId, float | list[float]]
    """Best drive amplitude for each qubit."""
    delta_amplitude: dict[QubitId, float | list[float]]
    """Difference in amplitude between native value and best fit."""
    rx90: bool
    """Pi or Pi_half calibration."""


FlippingAmplitudeType = np.dtype(
    [
        ("flips", np.float64),
        ("amplitude", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for flipping amplitude sweep."""


@dataclass
class FlippingAmplitudeData(Data):
    """FlippingAmplitude acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    pulse_amplitudes: dict[QubitId, float]
    """Native pulse amplitudes for each qubit."""
    rx90: bool
    """Pi or Pi_half calibration."""
    data: dict[QubitId, npt.NDArray[FlippingAmplitudeType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _acquisition(
    params: FlippingAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> FlippingAmplitudeData:
    r"""Data acquisition for flipping with amplitude sweep.

    For each combination of (flips, delta_amplitude) a sequence is built and
    executed.  The amplitude values are stored as absolute amplitudes
    (native + delta).  The resulting 2D map allows identifying the correct
    drive amplitude: at the true pi-pulse amplitude the excited-state
    probability should remain flat regardless of the number of flips.
    """

    data = FlippingAmplitudeData(
        resonator_type=platform.resonator_type,
        pulse_amplitudes={
            qubit: getattr(
                platform.natives.single_qubit[qubit], "RX90" if params.rx90 else "RX"
            )[0][1].amplitude
            for qubit in targets
        },
        rx90=params.rx90,
    )

    flips_sweep = range(0, params.nflips_max, params.nflips_step)
    delta_amplitude_sweep = np.arange(
        params.delta_amplitude_min,
        params.delta_amplitude_max,
        params.delta_amplitude_step,
    )

    sequences: list[PulseSequence] = []
    sweep_params: list[tuple[int, float]] = []

    for flips in flips_sweep:
        for delta_amp in delta_amplitude_sweep:
            sequence = PulseSequence()
            for qubit in targets:
                sequence += flipping_sequence(
                    platform=platform,
                    qubit=qubit,
                    delta_amplitude=float(delta_amp),
                    flips=flips,
                    rx90=params.rx90,
                )
            sequences.append(sequence)
            sweep_params.append((flips, float(delta_amp)))

    results = platform.execute(
        sequences,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )

    for (flips, delta_amp), sequence in zip(sweep_params, sequences):
        for qubit in targets:
            acq_channel = platform.qubits[qubit].acquisition
            assert acq_channel is not None
            ro_pulse = list(sequence.channel(acq_channel))[-1]
            assert isinstance(ro_pulse, Readout)
            prob = results[ro_pulse.id]
            error = np.sqrt(prob * (1 - prob) / params.nshots)
            native_amp = data.pulse_amplitudes[qubit]
            data.register_qubit(
                FlippingAmplitudeType,
                qubit,
                {
                    "flips": np.array([flips]),
                    "amplitude": np.array([native_amp + delta_amp]),
                    "prob": np.array([prob]),
                    "error": np.array([error]),
                },
            )

    return data


def _fit(data: FlippingAmplitudeData) -> FlippingAmplitudeResults:
    """Find the best amplitude by minimising the variance of P(|1>) vs flips.
    TODO: Use the same fit as in the flipping protocol to extract the rabi amp
    per input delta detiuning, these should all bve the same so the mean value
    can be used as the best amplitude.
    """

    best_amplitudes: dict[QubitId, list[float]] = {}
    delta_amplitudes: dict[QubitId, list[float]] = {}

    for qubit in data.qubits:
        qubit_data = data[qubit]
        amplitudes = np.unique(qubit_data["amplitude"])
        variances = []

        for amp in amplitudes:
            mask = qubit_data["amplitude"] == amp
            probs = qubit_data["prob"][mask]
            variances.append(float(np.var(probs)))

        best_idx = int(np.argmin(variances))
        best_amp = float(amplitudes[best_idx])
        native_amp = data.pulse_amplitudes[qubit]

        best_amplitudes[qubit] = [best_amp, 0.0]
        delta_amplitudes[qubit] = [best_amp - native_amp, 0.0]

    return FlippingAmplitudeResults(
        amplitude=best_amplitudes,
        delta_amplitude=delta_amplitudes,
        rx90=data.rx90,
    )


def _plot(
    data: FlippingAmplitudeData,
    target: QubitId,
    fit: FlippingAmplitudeResults | None = None,
):
    """Plotting function for FlippingAmplitude.

    Produces a heatmap of excited-state probability as a function of flips
    (x-axis) and pulse amplitude (y-axis).  When fit results are available a
    dashed horizontal line marks the best amplitude.
    """

    qubit_data = data[target]
    amplitudes = np.unique(qubit_data["amplitude"])
    flips_vals = np.unique(qubit_data["flips"])

    # Build 2D probability matrix: rows = amplitude, cols = flips
    z = np.full((len(amplitudes), len(flips_vals)), np.nan)
    amp_index = {amp: i for i, amp in enumerate(amplitudes)}
    flip_index = {fl: j for j, fl in enumerate(flips_vals)}

    for row in qubit_data:
        i = amp_index[row["amplitude"]]
        j = flip_index[row["flips"]]
        z[i, j] = row["prob"]

    fig = go.Figure(
        go.Heatmap(
            x=flips_vals,
            y=amplitudes,
            z=z,
            colorscale="RdBu",
            zmid=0.5,
            colorbar=dict(title="Excited State Probability"),
        )
    )

    if fit is not None and target in fit.amplitude:
        best_amp = fit.amplitude[target][0]
        fig.add_hline(
            y=best_amp,
            line=dict(color="black", dash="dash", width=2),
            annotation_text=f"Best amp: {best_amp:.4f}",
            annotation_position="right",
        )

    fig.update_layout(
        xaxis_title="Flips",
        yaxis_title="Amplitude [a.u.]",
    )

    fitting_report = ""
    if fit is not None and target in fit.amplitude:
        fitting_report = table_html(
            table_dict(
                target,
                ["Best amplitude [a.u.]", "Delta amplitude [a.u.]"],
                [fit.amplitude[target], fit.delta_amplitude[target]],
                display_error=True,
            )
        )

    return [fig], fitting_report


def _update(
    results: FlippingAmplitudeResults,
    platform: CalibrationPlatform,
    qubit: QubitId,
):
    update.drive_amplitude(results.amplitude[qubit], results.rx90, platform, qubit)


flipping_amplitude = Routine(_acquisition, _fit, _plot, _update)
"""FlippingAmplitude Routine object."""
