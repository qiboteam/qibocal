"""Protocol to sweep TWPA signal control parameters (frequency and offset) using sweepers."""

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    OscillatorConfig,
    Parameter,
    Platform,
    PulseSequence,
    Sweeper,
)

from ...auto.operation import Data, Parameters, Protocol, QubitId, Results
from ...result import magnitude
from ..utils import (
    HZ_TO_GHZ,
    RangeLike,
    readout_frequency,
    table_dict,
    table_html,
    to_range,
)


@dataclass
class TwpaFrequencyOffsetParameters(Parameters):
    """TwpaFrequencyOffset runcard inputs."""

    amplitude: RangeLike
    """Range of amplitude (offset) values for TWPA sweep."""
    frequency: RangeLike
    """Range of TWPA frequency values for sweep."""
    probes: list[float] | None = None
    """List of probe frequencies to evaluate (Hz). If omitted, defaults to readout frequencies of targets."""


@dataclass
class TwpaFrequencyOffsetResults(Results):
    """TwpaFrequencyOffset outputs."""

    data: dict[QubitId, npt.NDArray]
    """Array with average gain for each qubit."""
    frequency: dict[QubitId, float]
    """TWPA frequency [Hz]."""
    offset: dict[QubitId, float]
    """TWPA offset [dimensionless]."""


@dataclass
class TwpaFrequencyOffsetData(Data):
    """TwpaFrequencyOffset data acquisition."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""
    frequency: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa frequency values swept."""
    offset: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa offset values swept."""
    reference_value: dict[QubitId, list[float]] = field(default_factory=dict)
    """Reference values with TWPA off for each probe frequency."""
    probes: list[float] = field(default_factory=list)
    """List of probe frequencies evaluated."""
    attenuation: dict[QubitId, float] = field(default_factory=dict)
    """Configured base attenuation [dB] for each target."""

    def reference_value_array(self, qubit: QubitId) -> npt.NDArray:
        """Return reference value as a numpy array."""
        return np.array(self.reference_value[qubit]).reshape(-1, 2)

    def averaged_gain(self, qubit: QubitId) -> npt.NDArray:
        return 20 * np.log10(
            np.mean(magnitude(self[qubit]), axis=2)
            / np.mean(magnitude(self.reference_value_array(qubit)), axis=0)
        )


def _acquisition(
    params: TwpaFrequencyOffsetParameters,
    platform: Platform,
    targets: list[QubitId],
) -> TwpaFrequencyOffsetData:
    """Acquisition function for TwpaFrequencyOffset.

    First perform a scan over the readout probe with the TWPA off, then sweep the
    TWPA amplitude (offset) and frequency concurrently using a 2D sweeper.

    Note on the probes loop:
    The TWPA gain oscillates quite fast in the probe frequency. Instead of obtaining
    the best gain on average across an arbitrary continuous frequency range, we optimize
    specifically for the frequencies of the resonators we are expecting.
    Because not all control electronics support sweeping an arbitrary on-board list of
    values, we implement this as a software loop over the `probes` list. The main
    performance speed-up is already achieved through the on-board 2D hardware sweep over
    the TWPA pump signal parameters (amplitude and frequency).
    """
    sequence = PulseSequence()
    for qubit in targets:
        sequence += platform.natives.single_qubit[qubit].MZ()

    twpa_channels = {}
    for qubit in targets:
        pump = platform.channels[platform.qubits[qubit].acquisition].twpa_pump
        if pump is None:
            raise ValueError(
                f"Qubit {qubit} does not have a TWPA pump channel configured."
            )
        twpa_channels[qubit] = pump

    # Deduplicate TWPA channels preserving association to a target qubit
    unique_twpa_channels: dict[str, QubitId] = {}
    for q in targets:
        ch = twpa_channels[q]
        if ch not in unique_twpa_channels:
            unique_twpa_channels[ch] = q

    # Probe frequencies to evaluate
    probes = (
        params.probes
        if params.probes
        else [readout_frequency(q, platform) for q in targets]
    )

    # TWPA frequency ranges
    twpa_frequency_ranges = {
        q: np.arange(
            *to_range(
                params.frequency,
                center=platform.config(twpa_channels[q]).frequency,
            )
        ).tolist()
        for q in targets
    }

    # TWPA amplitude (offset) range (linear sweep)
    offset_range = to_range(params.amplitude)
    twpa_offset_values = np.arange(*offset_range)
    if np.any(np.abs(twpa_offset_values) >= 1.0):
        raise ValueError("TWPA amplitude values must be between -1 and 1.")
    twpa_offset_ranges = {q: twpa_offset_values.tolist() for q in targets}

    # Build 2D sweepers over TWPA pump parameters
    twpa_freq_sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=np.array(twpa_frequency_ranges[q]),
            channels=[ch],
        )
        for ch, q in unique_twpa_channels.items()
    ]
    twpa_offset_sweepers = [
        Sweeper(
            parameter=Parameter.offset,
            values=twpa_offset_values,
            channels=[ch],
        )
        for ch in unique_twpa_channels
    ]
    sweepers = [
        twpa_offset_sweepers,
        twpa_freq_sweepers,
    ]

    # Reference value acquisition (TWPA off)
    reference_data: dict[QubitId, list[list[float]]] = {q: [] for q in targets}
    raw_data: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}

    # 1. Reference measurements with TWPA off for each probe frequency
    try:
        for ch in unique_twpa_channels:
            if ch in platform.instruments:
                platform.instruments[ch].disconnect()

        for probe in probes:
            updates = [
                {ch: {"offset": 0.0} for ch in unique_twpa_channels}
                | {platform.qubits[q].probe: {"frequency": probe} for q in targets}
            ]
            ref_results = platform.execute(
                [sequence],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
                updates=updates,
            )
            for qubit in targets:
                acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[
                    -1
                ].id
                reference_data[qubit].append(ref_results[acq_handle].tolist())
    finally:
        for ch in unique_twpa_channels:
            if ch in platform.instruments:
                platform.instruments[ch].connect()

    # 2. 2D TWPA sweeps (amplitude and frequency) for each probe frequency
    for probe in probes:
        updates = [{platform.qubits[q].probe: {"frequency": probe}} for q in targets]
        results = platform.execute(
            [sequence],
            sweepers,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=updates,
        )
        for qubit in targets:
            acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[
                -1
            ].id
            raw_data[qubit].append(results[acq_handle])

    twpa_attenuations = {}
    for qubit in targets:
        cfg = cast(OscillatorConfig, platform.config(twpa_channels[qubit]))
        twpa_attenuations[qubit] = cfg.power

    data = TwpaFrequencyOffsetData(
        offset=twpa_offset_ranges,
        frequency=twpa_frequency_ranges,
        reference_value=reference_data,
        probes=probes,
        attenuation=twpa_attenuations,
    )
    for qubit in targets:
        data.data[qubit] = np.stack(raw_data[qubit], axis=2)

    return data


def _fit(data: TwpaFrequencyOffsetData) -> TwpaFrequencyOffsetResults:
    """Post-processing function for TwpaFrequencyOffset.

    After computing the averaged gain across evaluated probes, select the
    corresponding TWPA frequency and offset that maximizes the gain for each qubit.
    """
    gains = {}
    twpa_frequency = {}
    twpa_offset = {}
    for qubit in data.qubits:
        averaged_gain = data.averaged_gain(qubit)
        gains[qubit] = averaged_gain
        flat_index = np.argmax(averaged_gain)
        i, j = np.unravel_index(flat_index, averaged_gain.shape)
        twpa_frequency[qubit] = float(data.frequency[qubit][j])
        twpa_offset[qubit] = float(data.offset[qubit][i])
    return TwpaFrequencyOffsetResults(
        data=gains,
        frequency=twpa_frequency,
        offset=twpa_offset,
    )


def _plot(
    data: TwpaFrequencyOffsetData,
    fit: TwpaFrequencyOffsetResults | None,
    target: QubitId,
):
    """Plotting for TwpaFrequencyOffset."""
    figures = []
    fig = go.Figure()
    base_attenuation = data.attenuation.get(target, 0.0)

    averaged_gain = data.averaged_gain(target)
    offsets = np.array(data.offset[target])
    valid_mask = np.abs(offsets) > 1e-12
    tickvals = offsets[valid_mask]
    if len(tickvals) > 8:
        indices = np.linspace(0, len(tickvals) - 1, 8, dtype=int)
        tickvals = tickvals[indices]
    attenuations = base_attenuation + 20 * np.log10(np.abs(tickvals))
    ticktext = [f"{np.round(a, 1)}" for a in attenuations]

    fig.add_trace(
        go.Heatmap(
            x=np.array(data.frequency[target]) * HZ_TO_GHZ,
            y=data.offset[target],
            z=averaged_gain,
            colorscale="inferno",
        ),
    )
    if fit is not None and target in fit:
        fig.add_trace(
            go.Scatter(
                x=[fit.frequency[target] * HZ_TO_GHZ],
                y=[fit.offset[target]],
                mode="markers",
                marker={"size": 10, "color": "black", "symbol": "cross"},
                name="Optimal Point",
                showlegend=False,
            )
        )
    fig.update_xaxes(title_text="TWPA Frequency [GHz]")
    fig.update_yaxes(title_text="TWPA Amplitude")
    fig.update_layout(
        showlegend=False,
        yaxis2={
            "title_text": "TWPA Attenuation [dB]",
            "overlaying": "y",
            "side": "right",
            "matches": "y",
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "showgrid": False,
        },
    )

    figures.append(fig)

    if fit is not None and target in fit:
        opt_offset = fit.offset[target]
        opt_att = base_attenuation + 20 * np.log10(abs(opt_offset))
        labels = [
            "TWPA Frequency [Hz]",
            "TWPA Amplitude",
            "TWPA Attenuation [dB]",
        ]
        values = [
            np.round(fit.frequency[target], 4),
            np.round(fit.offset[target], 4),
            np.round(opt_att, 4),
        ]
        fitting_report = table_html(
            table_dict(
                [target] * len(labels),
                labels,
                values,
            )
        )
    else:
        fitting_report = ""

    return figures, fitting_report


twpa_frequency_offset = Protocol(_acquisition, _fit, _plot)
"""Resonator TWPA Frequency and Offset Sweeper Protocol object."""

twpa_sweep = twpa_frequency_offset
"""Alias for twpa_frequency_offset."""
