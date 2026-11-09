"""Protocol to calibrate TWPA power and frequency for a specific probe frequency."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    Acquisition,
    AcquisitionChannel,
    AcquisitionType,
    AveragingMode,
    OscillatorConfig,
    Parameter,
    Pulse,
    PulseSequence,
    Qubit,
    Readout,
    Rectangular,
    Sweeper,
)
from qibolab._core.instruments.oscillator import LocalOscillator

from ...auto.operation import Data, Parameters, Protocol, QubitId, Results
from ...calibration.platform import CalibrationPlatform
from ...result import magnitude
from ..utils import (
    HZ_TO_GHZ,
    Range,
    RangeLike,
    readout_frequency,
    table_dict,
    table_html,
    to_range,
)


@dataclass
class TwpaCalibrationParameters(Parameters):
    """TwpaCalibration runcard inputs."""

    freq_width: float | None = None
    """Width for frequency sweep of readout pulse (Hz)."""
    freq_step: float | None = None
    """Frequency step for sweep of readout pulse (Hz)."""
    probe_frequency: RangeLike | None = None
    """Frequency [Hz] range for sweep."""
    probe_duration: float = 8e3
    """Duration [ns] of the rectangular probe, and of its acquisition."""
    probe_amplitude: float = 0.1
    """Amplitude of the rectangular probe."""
    twpa_freq_width: float | None = None
    """Width for TPWA frequency sweep (Hz)."""
    twpa_freq_step: float | None = None
    """TPWA frequency step (Hz)."""
    frequency: RangeLike | None = None
    """TWPA pump frequency [Hz] range for sweep."""
    twpa_pow_width: float | None = None
    """Width for TPWA power sweep (dBm)."""
    twpa_pow_step: float | None = None
    """TPWA power step (dBm)."""
    power: RangeLike | None = None
    """TWPA pump power [dBm] range for sweep."""

    def probe_frequency_range(self, center: float = 0.0) -> Range:
        def legacy_range() -> Range:
            if self.freq_width is None or self.freq_step is None:
                raise ValueError(
                    "Provide 'probe_frequency' or both 'freq_width' and 'freq_step'."
                )
            return (
                center - self.freq_width / 2,
                center + self.freq_width / 2,
                self.freq_step,
            )

        return (
            to_range(self.probe_frequency, center=center)
            if self.probe_frequency is not None
            else legacy_range()
        )

    def frequency_range(self, center: float = 0.0) -> Range:
        def legacy_range() -> Range:
            if self.twpa_freq_width is None or self.twpa_freq_step is None:
                raise ValueError(
                    "Provide 'frequency' or both 'twpa_freq_width' and 'twpa_freq_step'."
                )
            return (
                center - self.twpa_freq_width / 2,
                center + self.twpa_freq_width / 2,
                self.twpa_freq_step,
            )

        return (
            to_range(self.frequency, center=center)
            if self.frequency is not None
            else legacy_range()
        )

    def power_range(self, center: float = 0.0) -> Range:
        def legacy_range() -> Range:
            if self.twpa_pow_width is None or self.twpa_pow_step is None:
                raise ValueError(
                    "Provide 'power' or both 'twpa_pow_width' and 'twpa_pow_step'."
                )
            return (
                center - self.twpa_pow_width / 2,
                center + self.twpa_pow_width / 2,
                self.twpa_pow_step,
            )

        return (
            to_range(self.power, center=center)
            if self.power is not None
            else legacy_range()
        )


@dataclass
class TwpaCalibrationResults(Results):
    """TwpaCalibration outputs."""

    data: dict[QubitId, npt.NDArray[np.float64]]
    """Array with average gain for each qubit."""
    twpa_frequency: dict[QubitId, float]
    """TWPA frequency [Hz]."""
    twpa_power: dict[QubitId, float]
    """TWPA power [dBm]."""


@dataclass
class TwpaCalibrationData(Data):
    """TwpaCalibration data acquisition."""

    data: dict[QubitId, npt.NDArray[np.float64]] = field(default_factory=dict)
    """Raw data acquired."""
    twpa_frequency: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa frequency values swept."""
    twpa_power: dict[QubitId, list[float]] = field(default_factory=dict)
    """List with twpa power values swept."""
    reference_value: dict[QubitId, list[float]] = field(default_factory=dict)
    """Values for readout frequency sweep with TWPA off."""

    def reference_value_array(self, qubit: QubitId) -> npt.NDArray[np.float64]:
        """Return reference value as a numpy array."""
        return np.array(self.reference_value[qubit]).reshape(-1, 2)


def _sweep(range_: Range) -> list[float]:
    """Values of a range, as a list.

    The number of values only depends on the width of the range, and not on its
    center, unlike :func:`numpy.arange`, which suffers of floating point errors on the
    end points. Thus, ranges with the same width and step, but different centers, have
    the same number of values.
    """
    start, stop, step = range_
    # as in numpy.arange, the stop is excluded, while rounding the number of steps
    # discards floating point errors in the width
    count = max(int(np.ceil(np.round((stop - start) / step, 9))), 0)
    return (start + step * np.arange(count)).tolist()


def _reference_scan(
    platform: CalibrationPlatform,
    sequence: PulseSequence,
    sweepers: list[Sweeper],
    nshots: int,
    relaxation_time: float,
    acquisition_channels: Sequence[str],
    twpas: Sequence[LocalOscillator],
) -> list[list[float]]:
    """Reference scan with TWPA off to compute the gain later on."""
    reference_value: list[list[float]] = []
    # reference value without twpas, turning them back on even in case of errors
    try:
        for twpa in twpas:
            assert twpa.device is not None
            twpa.device.off()

        results = platform.execute(
            [sequence],
            [sweepers],
            nshots=nshots,
            relaxation_time=relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
    finally:
        for twpa in twpas:
            assert twpa.device is not None
            twpa.device.on()

    for ch in acquisition_channels:
        acq_handle = list(sequence.channel(ch))[-1].id
        reference_value.append(results[acq_handle].tolist())
    return reference_value


def _twpa_scan(
    platform: CalibrationPlatform,
    sequence: PulseSequence,
    sweepers: list[Sweeper],
    qubits: Sequence[QubitId],
    pumps: Sequence[str],
    acquisition_channels: Sequence[str],
    params: TwpaCalibrationParameters,
) -> TwpaCalibrationData:
    """TWPA scan to compute the gain."""
    twpa_configs = {
        pump: cast(OscillatorConfig, platform.config(pump)) for pump in pumps
    }
    # pumps are scanned simultaneously, thus their ranges must have the same size
    power_ranges = {
        pump: _sweep(params.power_range(twpa.power))
        for pump, twpa in twpa_configs.items()
    }
    frequency_ranges = {
        pump: _sweep(params.frequency_range(twpa.frequency))
        for pump, twpa in twpa_configs.items()
    }
    for name, ranges in (("power", power_ranges), ("frequency", frequency_ranges)):
        if len({len(values) for values in ranges.values()}) > 1:
            raise ValueError(
                f"TWPA pumps' {name} ranges must have the same size, "
                f"found {[len(values) for values in ranges.values()]}."
            )

    data = TwpaCalibrationData(
        twpa_power={q: power_ranges[pump] for q, pump in zip(qubits, pumps)},
        twpa_frequency={q: frequency_ranges[pump] for q, pump in zip(qubits, pumps)},
    )

    data_: dict[QubitId, list[npt.NDArray[np.float64]]] = defaultdict(list)
    for powers in zip(*power_ranges.values()):
        for frequencies in zip(*frequency_ranges.values()):
            updates: list[dict[str, dict[str, Any]]] = []
            for twpa, power, frequency in zip(twpa_configs.keys(), powers, frequencies):
                updates.append({twpa: {"power": power, "frequency": frequency}})
            results = platform.execute(
                [sequence],
                [sweepers],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
                updates=updates,
            )
            for qubit, ch in zip(qubits, acquisition_channels):
                acq_handle = list(sequence.channel(ch))[-1].id
                data_[qubit].append(results[acq_handle])

    data.data = {
        qubit: np.stack(data_[qubit], axis=0).reshape(
            len(data.twpa_power[qubit]),
            len(data.twpa_frequency[qubit]),
            len(sweeper.values),
            2,
        )
        for qubit, sweeper in zip(qubits, sweepers)
    }

    return data


def _acquisition(
    params: TwpaCalibrationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TwpaCalibrationData:
    """Acquisition function for TwpaCalibration.

    First perform a scan over the readout probe with the TWPA off, then we sweep the TWPA power and frequency.
    The gain is computed as the norm of the complex readout signal divided the norm of the complex readout signal without TWPA.
    """

    qubits: list[Qubit] = [platform.qubits[qubit] for qubit in targets]
    acquisition_channels: list[str] = []
    pumps: list[str] = []
    for target, qubit in zip(targets, qubits):
        if qubit.acquisition is None:
            raise ValueError(f"Acquisition channel for qubit {target} not defined.")
        acquisition_channels.append(qubit.acquisition)
        pump = cast(AcquisitionChannel, platform.channels[qubit.acquisition]).twpa_pump
        if pump is None:
            raise ValueError(
                f"Qubit {target} does not have a TWPA pump channel configured."
            )
        pumps.append(pump)

    sequence = PulseSequence()
    for ch in acquisition_channels:
        sequence += [
            (
                ch,
                Readout(
                    acquisition=Acquisition(duration=params.probe_duration),
                    probe=Pulse(
                        duration=params.probe_duration,
                        amplitude=params.probe_amplitude,
                        envelope=Rectangular(),
                    ),
                ),
            )
        ]

    # probes are swept in parallel, thus their sweeps must have the same size
    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=np.array(
                _sweep(params.probe_frequency_range(readout_frequency(q, platform)))
            ),
            channels=[qubit.probe],
        )
        for q, qubit in zip(targets, qubits)
    ]
    reference_value = _reference_scan(
        platform,
        sequence,
        sweepers,
        params.nshots,
        params.relaxation_time,
        acquisition_channels,
        # multiplexed qubits may share a pump: each one is turned off only once
        [
            cast(LocalOscillator, platform.instruments[pump])
            for pump in dict.fromkeys(pumps)
        ],
    )

    data = _twpa_scan(
        platform, sequence, sweepers, targets, pumps, acquisition_channels, params
    )
    data.reference_value = dict(zip(targets, reference_value))
    return data


def _fit(data: TwpaCalibrationData) -> TwpaCalibrationResults:
    """Post-processing function for TwpaCalibration.

    After computing the averaged gain we select the corresponding twpa frequency and power
    that maximizes the gain for each qubit.
    """
    gains: dict[QubitId, npt.NDArray[np.float64]] = {}
    twpa_frequency: dict[QubitId, float] = {}
    twpa_power: dict[QubitId, float] = {}
    for qubit in data.qubits:
        averaged_gain = 20 * np.log10(
            np.mean(magnitude(data[qubit]), axis=2)
            / np.mean(magnitude(data.reference_value_array(qubit)), axis=0)
        )
        gains[qubit] = averaged_gain
        flat_index = np.argmax(averaged_gain)
        i, j = np.unravel_index(flat_index, averaged_gain.shape)
        twpa_frequency[qubit] = data.twpa_frequency[qubit][j]
        twpa_power[qubit] = data.twpa_power[qubit][i]
    return TwpaCalibrationResults(
        data=gains,
        twpa_frequency=twpa_frequency,
        twpa_power=twpa_power,
    )


def _plot(
    data: TwpaCalibrationData, fit: TwpaCalibrationResults | None, target: QubitId
) -> tuple[list[go.Figure], str]:
    """Plotting for TwpaCalibration."""

    figures: list[go.Figure] = []
    fig = go.Figure()
    if fit is not None:
        fitting_report = table_html(
            table_dict(
                [target, target],
                [
                    "TWPA Frequency [Hz]",
                    "TWPA Power [dBm]",
                ],
                [
                    np.round(fit.twpa_frequency[target], 4),
                    np.round(fit.twpa_power[target], 4),
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
            x=np.array(data.twpa_frequency[target]) * HZ_TO_GHZ,
            y=data.twpa_power[target],
            z=averaged_gain,
            colorscale="inferno",
        ),
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="TWPA Frequency [GHz]",
        yaxis_title="TWPA Power [dBm]",
    )

    figures.append(fig)

    return figures, fitting_report


twpa_calibration = Protocol(_acquisition, _fit, _plot)
"""Resonator TWPA Frequency Protocol object."""
