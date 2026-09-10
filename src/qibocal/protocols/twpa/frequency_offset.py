"""Protocol to sweep TWPA signal control parameters (frequency and offset) using sweepers."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
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
    probe_frequency: RangeLike
    """Range of readout probe frequency values for sweep."""


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
    """List with twpa frequency values swept, for each qubit."""
    offset: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa offset values swept, for each qubit."""
    probes: dict[QubitId, list[float]] = field(default_factory=dict)
    """Values for readout frequency sweep with TWPA off, for each qubit."""

    def reference_value_array(self, qubit: QubitId) -> npt.NDArray:
        """Return reference value as a numpy array."""
        return np.array(self.probes[qubit]).reshape(-1, 2)


def _acquisition(
    params: TwpaFrequencyOffsetParameters,
    platform: Platform,
    targets: list[QubitId],
) -> TwpaFrequencyOffsetData:
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

    # Readout probe frequency ranges
    ro_freq_values = {
        q: np.arange(
            *to_range(
                params.probe_frequency,
                center=readout_frequency(q, platform),
            )
        )
        for q in targets
    }

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

    # Build sweepers
    ro_freq_sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=ro_freq_values[q],
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]
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

    # Reference value acquisition (TWPA off)
    zero_offset_sweepers = [
        Sweeper(
            parameter=Parameter.offset,
            values=np.array([0.0]),
            channels=[ch],
        )
        for ch in unique_twpa_channels
    ]
    ref_sweepers = [zero_offset_sweepers, ro_freq_sweepers]

    reference_value = {}
    try:
        for ch in unique_twpa_channels:
            if ch in platform.instruments:
                platform.instruments[ch].disconnect()

        ref_results = platform.execute(
            [sequence],
            ref_sweepers,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
    finally:
        for ch in unique_twpa_channels:
            if ch in platform.instruments:
                platform.instruments[ch].connect()

    for qubit in targets:
        acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[-1].id
        reference_value[qubit] = ref_results[acq_handle].tolist()

    data = TwpaFrequencyOffsetData(
        offset=twpa_offset_ranges,
        frequency=twpa_frequency_ranges,
        probes=reference_value,
    )

    # Main scan: sweep offset, twpa frequency, and readout frequency concurrently
    sweepers = [
        twpa_offset_sweepers,
        twpa_freq_sweepers,
        ro_freq_sweepers,
    ]
    results = platform.execute(
        [sequence],
        sweepers,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for qubit in targets:
        acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[-1].id
        data.data[qubit] = np.array(results[acq_handle])

    return data


def _fit(data: TwpaFrequencyOffsetData) -> TwpaFrequencyOffsetResults:
    """Maximize gain (bare argmax).

    After computing the averaged gain we select the corresponding TWPA frequency and offset
    that maximizes the gain for each qubit.
    """
    gains = {}
    twpa_frequency = {}
    twpa_offset = {}
    for qubit in data.qubits:
        averaged_gain = 20 * np.log10(
            np.mean(magnitude(data[qubit]), axis=2)
            / np.mean(magnitude(data.reference_value_array(qubit)), axis=0)
        )
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
    figures = []
    fig = go.Figure()
    if fit is not None and target in fit:
        fitting_report = table_html(
            table_dict(
                [target, target],
                [
                    "TWPA Frequency [Hz]",
                    "TWPA Amplitude",
                ],
                [
                    np.round(fit.frequency[target], 4),
                    np.round(fit.offset[target], 4),
                ],
            )
        )
        averaged_gain = fit.data[target]
    else:
        averaged_gain = 20 * np.log10(
            np.mean(magnitude(data[target]), axis=2)
            / np.mean(magnitude(data.reference_value_array(target)), axis=0)
        )
        fitting_report = ""

    fig.add_trace(
        go.Heatmap(
            x=np.array(data.frequency[target]) * HZ_TO_GHZ,
            y=data.offset[target],
            z=averaged_gain,
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
    )

    figures.append(fig)

    return figures, fitting_report


twpa_frequency_offset = Protocol(_acquisition, _fit, _plot)
"""Resonator TWPA Frequency and Offset Sweeper Protocol object.

First perform a scan over the readout probe with the TWPA off, then sweep the
TWPA amplitude (offset) and frequency concurrently with the readout probe using sweepers.
The gain is computed as the norm of the complex readout signal divided by the
norm of the complex readout signal without TWPA.
"""

twpa_sweep = twpa_frequency_offset
"""Alias for twpa_frequency_offset."""
