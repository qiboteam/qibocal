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
    probe_amplitude: float = 0.1
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
            assert self.freq_width is not None and self.freq_step is not None
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
            assert self.twpa_freq_width is not None and self.twpa_freq_step is not None
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
            assert self.twpa_pow_width is not None and self.twpa_pow_step is not None
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
    """TWPA frequency [GHz]."""
    twpa_power: dict[QubitId, float]
    """TWPA power [dBm]."""


@dataclass
class TwpaCalibrationData(Data):
    """TwpaCalibration data acquisition."""

    data: dict[QubitId, npt.NDArray[np.float64]] = field(default_factory=dict)
    """Raw data acquired."""
    twpa_frequency: dict[QubitId, list[float]] = field(default=dict)
    """List with twpa frequency values swept."""
    twpa_power: dict[QubitId, list[float]] = field(default=dict)
    """List with twpa power values swept."""
    reference_value: dict[QubitId, list[float]] = field(default=dict)
    """Values for readout frequency sweep with TWPA off."""

    def reference_value_array(self, qubit: QubitId) -> npt.NDArray[np.float64]:
        """Return reference value as a numpy array."""
        return np.array(self.reference_value[qubit]).reshape(-1, 2)


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
    # reference value without twpas
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
    for ch in acquisition_channels:
        acq_handle = list(sequence.channel(ch))[-1].id
        reference_value.append(results[acq_handle].tolist())

    for twpa in twpas:
        assert twpa.device is not None
        twpa.device.on()
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
    power_ranges: dict[QubitId, list[float]] = {
        q: np.arange(*params.power_range(twpa.power)).tolist()
        for q, twpa in zip(qubits, twpa_configs.values())
    }
    frequency_ranges: dict[QubitId, list[float]] = {
        q: np.arange(*params.frequency_range(twpa.frequency)).tolist()
        for q, twpa in zip(qubits, twpa_configs.values())
    }

    data = TwpaCalibrationData(
        twpa_power=power_ranges,
        twpa_frequency=frequency_ranges,
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
            len(np.arange(*params.power_range())),
            len(np.arange(*params.frequency_range())),
            len(np.arange(*params.probe_frequency_range())),
            2,
        )
        for qubit in qubits
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
    for qubit in qubits:
        assert qubit.acquisition is not None
        acquisition_channels.append(qubit.acquisition)
        pump = cast(AcquisitionChannel, platform.channels[qubit.acquisition]).twpa_pump
        assert pump is not None
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

    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            range=params.probe_frequency_range(readout_frequency(q, platform)),
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
        [cast(LocalOscillator, platform.instruments[pump]) for pump in pumps],
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


def _plot(data: TwpaCalibrationData, fit: TwpaCalibrationResults, target):
    """Plotting for TwpaCalibration."""

    figures = []
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
