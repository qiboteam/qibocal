"""Protocol to sweep TWPA signal control parameters (frequency and offset) using sweepers."""

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    Acquisition,
    AcquisitionType,
    AveragingMode,
    ChannelId,
    OscillatorConfig,
    Parameter,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    Sweeper,
)
from scipy.constants import nano

from ...auto.operation import Data, Parameters, Protocol, QubitId, Results
from ...calibration.platform import CalibrationPlatform
from ...result import magnitude
from ..utils import (
    RangeLike,
    readout_frequency,
    table_dict,
    table_html,
    to_range,
)


@dataclass
class TwpaFrequencyOffsetParameters(Parameters):
    amplitude: RangeLike
    """Range of amplitude (offset) values for TWPA sweep."""
    frequency: RangeLike
    """Range of TWPA frequency values for sweep."""
    probes: list[float] | None = None
    """List of probe frequencies to evaluate (Hz).

    If omitted, a single acquisition run is performed where each target is probed
    exclusively at its own calibrated readout frequency.
    """
    probe_duration: float = 4e3
    """Probe wave duration."""
    probe_amplitude: float = 1.0
    """Probe wave amplitude."""


@dataclass
class TwpaFrequencyOffsetResults(Results):
    frequency: dict[QubitId, float]
    """Pump frequency [Hz]."""
    offset: dict[QubitId, float]
    """Pump offset [a.u.]."""
    gain: dict[QubitId, float]
    """TWPA gain [dB]."""


@dataclass
class TwpaFrequencyOffsetData(Data):
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""
    frequency: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa frequency values swept."""
    offset: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa offset values swept."""
    reference_value: dict[QubitId, list[list[float]]] = field(default_factory=dict)
    """Reference values with TWPA off for each probe frequency.

    Reference measurements are I/Q points themselves, as the acquired data.
    """
    probes: list[float] | None = None
    """List of probe frequencies evaluated.

    ``None`` if native readout frequencies were used.
    """
    attenuation: dict[QubitId, float] = field(default_factory=dict)
    """Configured base attenuation [dB] for each target."""

    def reference_value_array(self, qubit: QubitId) -> npt.NDArray:
        """Return reference value as a numpy array.

        The shape is ``(nprobes, 2)``, with ``nprobes`` the number of probes used, and 2
        for the I/Q measurements.
        """
        return np.array(self.reference_value[qubit]).reshape(-1, 2)

    def averaged_gain(self, qubit: QubitId) -> npt.NDArray:
        return 20 * np.log10(
            np.mean(magnitude(self[qubit]), axis=2)
            / np.mean(magnitude(self.reference_value_array(qubit)), axis=0)
        )


def _acquisition(
    params: TwpaFrequencyOffsetParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TwpaFrequencyOffsetData:
    """Acquire reference scan and 2D gain landscape, for each probe.

    First perform a scan over the readout probe with the TWPA off, then sweep the
    TWPA amplitude (offset) and frequency concurrently using a 2D sweeper.

    Note on the probe frequencies handling:
    The TWPA gain oscillates quite fast in the probe frequency. Instead of obtaining
    the best gain on average across an arbitrary continuous frequency range, we optimize
    specifically for the frequencies of the expected readout resonators.
    When `params.probes` is omitted (None), the protocol performs a single acquisition run
    where each target qubit is updated and probed exclusively at its own calibrated
    readout frequency.
    When `params.probes` is provided as a discrete list of frequencies, the protocol
    iterates over each probe frequency in software, applying it to all targets across
    successive runs. Because not all control electronics support sweeping an arbitrary
    on-board list of values, this multi-probe evaluation is implemented as a software loop.
    The primary speed-up is achieved through the on-board 2D hardware sweep over
    the TWPA pump signal parameters (amplitude and frequency).
    """
    acquisition: dict[QubitId, ChannelId] = {}
    for q in targets:
        acq = platform.qubits[q].acquisition
        if acq is None:
            raise ValueError(f"Acquisition channel for qubit {q} not defined.")
        acquisition[q] = acq

    # The sequence is purely made by simultaneous rectangular readouts, whose duration
    # and amplitude are given as inputs
    readouts = {
        q: Readout(
            probe=Pulse(
                amplitude=params.probe_amplitude,
                duration=params.probe_duration,
                envelope=Rectangular(),
            ),
            acquisition=Acquisition(duration=params.probe_duration),
        )
        for q in targets
    }
    sequence = PulseSequence([(acquisition[q], readouts[q]) for q in targets])
    acquisition_handles = {q: readouts[q].acquisition.id for q in targets}

    twpa_channels = {}
    twpa_configs = {}
    for qubit in targets:
        pump = platform.channels[platform.qubits[qubit].acquisition].twpa_pump
        if pump is None:
            raise ValueError(
                f"Qubit {qubit} does not have a TWPA pump channel configured."
            )
        twpa_channels[qubit] = pump
        twpa_configs[qubit] = cast(OscillatorConfig, platform.config(pump))

    # Deduplicate TWPA channels preserving association to a target qubit
    unique_twpa_channels: dict[str, QubitId] = {}
    for q in targets:
        ch = twpa_channels[q]
        if ch not in unique_twpa_channels:
            unique_twpa_channels[ch] = q

    # Probe configurations to evaluate
    if params.probes is not None:
        probe_updates_list = [
            {platform.qubits[q].probe: {"frequency": probe} for q in targets}
            for probe in params.probes
        ]
    else:
        probe_updates_list = [
            {
                platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}
                for q in targets
            }
        ]

    # TWPA frequency ranges
    freq_ranges = {
        q: to_range(params.frequency, center=twpa_configs[q].frequency) for q in targets
    }

    # TWPA amplitude (offset) range (linear sweep)
    offset_range = to_range(params.amplitude)

    # Build 2D sweepers over TWPA pump parameters.
    # Use Sweeper-generated values to ensure consistency with backend semantics.
    freq_sweeps = [
        Sweeper(
            parameter=Parameter.frequency,
            range=freq_ranges[q],
            channels=[ch],
        )
        for ch, q in unique_twpa_channels.items()
    ]
    offset_sweeps = [
        Sweeper(
            parameter=Parameter.offset,
            range=offset_range,
            channels=[ch],
        )
        for ch in unique_twpa_channels
    ]
    offset_values = offset_sweeps[0].values
    if np.any(np.abs(offset_values) >= 1.0):
        raise ValueError("TWPA amplitude values must be between -1 and 1.")

    # Reference value acquisition (TWPA off)
    reference_data: dict[QubitId, list[list[float]]] = {q: [] for q in targets}
    raw_data: dict[QubitId, list[npt.NDArray]] = {q: [] for q in targets}

    # 1. Reference measurements with TWPA off
    for probe_updates in probe_updates_list:
        updates = [{ch: {"offset": 0.0} for ch in unique_twpa_channels} | probe_updates]
        ref_results = platform.execute(
            [sequence],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=updates,
        )
        for qubit in targets:
            reference_data[qubit].append(
                ref_results[acquisition_handles[qubit]].tolist()
            )

    # 2. 2D TWPA sweeps (amplitude and frequency)
    for probe_updates in probe_updates_list:
        updates = [probe_updates]
        results = platform.execute(
            [sequence],
            [offset_sweeps, freq_sweeps],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=updates,
        )
        for qubit in targets:
            raw_data[qubit].append(results[acquisition_handles[qubit]])

    data = TwpaFrequencyOffsetData(
        offset={q: offset_values.tolist() for q in targets},
        frequency={q: np.arange(*freq_ranges[q]).tolist() for q in targets},
        reference_value=reference_data,
        probes=params.probes,
        attenuation={q: twpa_configs[q].power for q in targets},
    )
    for qubit in targets:
        data.data[qubit] = np.stack(raw_data[qubit], axis=2)

    return data


def _fit(data: TwpaFrequencyOffsetData) -> TwpaFrequencyOffsetResults:
    """Maximize measured gain.

    After computing the averaged gain across evaluated probes, select the
    corresponding TWPA frequency and offset that maximizes the gain for each qubit.
    """
    gains = {}
    frequency = {}
    offset = {}
    gain = {}
    for qubit in data.qubits:
        averaged_gain = data.averaged_gain(qubit)
        gains[qubit] = averaged_gain
        flat_index = np.argmax(averaged_gain)
        i, j = np.unravel_index(flat_index, averaged_gain.shape)
        frequency[qubit] = data.frequency[qubit][j]
        offset[qubit] = data.offset[qubit][i]
        gain[qubit] = averaged_gain[i, j]
    return TwpaFrequencyOffsetResults(frequency=frequency, offset=offset, gain=gain)


def _plot(
    data: TwpaFrequencyOffsetData,
    fit: TwpaFrequencyOffsetResults | None,
    target: QubitId,
):
    """Plot average gain, and report fit results - if any.

    The visualization displays the averaged TWPA gain across evaluated probe frequencies
    as a 2D heatmap versus pump frequency (horizontal axis) and pump amplitude/offset
    (primary vertical axis). If fit results are available, the optimal working point is
    highlighted with a marker in the plot and legend.

    To relate the dimensionless amplitude offset to the effective physical attenuation
    in dB, a secondary vertical axis is rendered on the right edge. Its tick positions
    and labels are generated dynamically from the platform attenuation and the logarithmic
    offset scaling. Because Plotly requires an associated trace to render secondary layout
    axes, an invisible dummy scatter trace is attached to this secondary scale.
    """
    figures = []
    fig = go.Figure()
    base_attenuation = data.attenuation.get(target, 0.0)

    averaged_gain = data.averaged_gain(target)
    offsets = np.array(data.offset[target])
    frequencies = np.array(data.frequency[target]) * nano
    valid_mask = np.abs(offsets) > 1e-12
    tickvals = offsets[valid_mask]
    if len(tickvals) > 8:
        indices = np.linspace(0, len(tickvals) - 1, 8, dtype=int)
        tickvals = tickvals[indices]
    attenuations = base_attenuation + 20 * np.log10(np.abs(tickvals))
    ticktext = [f"{np.round(a, 1)}" for a in attenuations]

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=data.offset[target],
            z=averaged_gain,
            colorscale="inferno",
            colorbar_x=1.15,
        ),
    )
    # Invisible trace required for Plotly to render the secondary y-axis (yaxis2).
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 0, "opacity": 0},
            hoverinfo="skip",
            showlegend=False,
            yaxis="y2",
        )
    )
    if fit is not None and target in fit:
        fig.add_trace(
            go.Scatter(
                x=[fit.frequency[target] * nano],
                y=[fit.offset[target]],
                mode="markers",
                marker={"size": 10, "color": "black", "symbol": "cross"},
                name="Optimal Point",
                showlegend=True,
            )
        )
    fig.update_xaxes(title_text="Pump Frequency [GHz]")
    fig.update_yaxes(title_text="Pump Amplitude")
    if len(frequencies) > 1:
        df = abs(frequencies[1] - frequencies[0]) / 2
        fig.update_xaxes(range=[np.min(frequencies) - df, np.max(frequencies) + df])
    if len(offsets) > 1:
        doff = abs(offsets[1] - offsets[0]) / 2
        fig.update_yaxes(range=[np.min(offsets) - doff, np.max(offsets) + doff])
    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h"},
        yaxis2={
            "title_text": "Pump Attenuation [dB]",
            "overlaying": "y",
            "side": "right",
            "matches": "y",
            "tickmode": "array",
            "tickvals": tickvals.tolist(),
            "ticktext": ticktext,
            "showgrid": False,
        },
    )

    figures.append(fig)

    if fit is not None and target in fit:
        opt_offset = fit.offset[target]
        opt_att = base_attenuation + 20 * np.log10(abs(opt_offset))
        labels = [
            "Pump Frequency [Hz]",
            "Pump Amplitude",
            "Pump Attenuation [dB]",
            "TWPA Gain [dB]",
        ]
        values = [
            np.round(fit.frequency[target], 4),
            np.round(opt_offset, 4),
            np.round(opt_att, 4),
            np.round(fit.gain[target], 4),
        ]
        fitting_report = table_html(table_dict([target] * len(labels), labels, values))
    else:
        fitting_report = ""

    return figures, fitting_report


twpa_frequency_offset = Protocol(_acquisition, _fit, _plot)
"""Resonator TWPA Frequency and Offset Sweeper Protocol object."""

twpa_sweep = twpa_frequency_offset
"""Alias for twpa_frequency_offset."""
