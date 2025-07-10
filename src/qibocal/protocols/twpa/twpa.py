"""Protocol to calibrate TWPA power and frequency for a specific probe frequency."""

from dataclasses import dataclass, field
from typing import Optional

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

from ...auto.operation import Data, Parameters, QubitId, Results, Routine
from ...result import magnitude
from ..utils import HZ_TO_GHZ, readout_frequency, table_dict, table_html


@dataclass
class TwpaCalibrationParameters(Parameters):
    """TwpaCalibration runcard inputs."""

    freq_width: float
    """Width for frequency sweep of readout pulse (Hz)."""
    freq_step: float
    """Frequency step for sweep of readout pulse (Hz)."""
    twpa_freq_width: int
    """Width for TPWA frequency sweep (Hz)."""
    twpa_freq_step: int
    """TPWA frequency step (Hz)."""
    twpa_pow_width: int
    """Width for TPWA power sweep (dBm)."""
    twpa_pow_step: int
    """TPWA power step (dBm)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class TwpaCalibrationResults(Results):
    """TwpaCalibration outputs."""

    data: dict[QubitId, npt.NDArray]
    """Array with average gain for each qubit."""
    twpa_frequency: dict[QubitId, float]
    """TWPA frequency [GHz]."""
    twpa_power: dict[QubitId, float]
    """TWPA power [dBm]."""


@dataclass
class TwpaCalibrationData(Data):
    """TwpaCalibration data acquisition."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""
    twpa_frequency: dict[QubitId, list[float]] = field(default=list)
    """List with twpa frequency values swept."""
    twpa_power: dict[QubitId, list[float]] = field(default=list)
    """List with twpa power values swept."""
    reference_value: dict[QubitId, list[float]] = field(default=list)
    """Values for readout frequency sweep with TWPA off."""

    def reference_value_array(self, qubit: QubitId) -> npt.NDArray:
        """Return reference value as a numpy array."""
        return np.array(self.reference_value[qubit]).reshape(-1, 2)


def _acquisition(
    params: TwpaCalibrationParameters,
    platform: Platform,
    targets: list[QubitId],
) -> TwpaCalibrationData:
    """Acquisition function for TwpaCalibration.

    First perform a scan over the readout probe with the TWPA off, then we sweep the TWPA power and frequency.
    The gain is computed as the norm of the complex readout signal divided the norm of the complex readout signal without TWPA.
    """

    sequence = PulseSequence()
    for qubit in targets:
        sequence += platform.natives.single_qubit[qubit].MZ()

    twpas = {
        platform.channels[
            platform.qubits[qubit].acquisition
        ].twpa_pump: platform.instruments[
            platform.channels[platform.qubits[qubit].acquisition].twpa_pump
        ]
        for qubit in targets
    }

    frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    twpa_frequency_range = np.arange(
        -params.twpa_freq_width / 2,
        params.twpa_freq_width / 2,
        params.twpa_freq_step,
    )
    twpa_power_range = np.arange(
        -params.twpa_pow_width / 2, params.twpa_pow_width / 2, params.twpa_pow_step
    )

    twpa_power_ranges = {
        qubit: (
            twpa_power_range
            + platform.config(
                platform.channels[platform.qubits[qubit].acquisition].twpa_pump
            ).power
        ).tolist()
        for qubit in targets
    }
    twpa_frequency_ranges = {
        qubit: (
            twpa_frequency_range
            + platform.config(
                platform.channels[platform.qubits[qubit].acquisition].twpa_pump
            ).frequency
        ).tolist()
        for qubit in targets
    }
    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=readout_frequency(q, platform) + frequency_range,
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]
    reference_value = {}
    # reference value without twpas
    for twpa in twpas.values():
        twpa.disconnect()

    results = platform.execute(
        [sequence],
        [sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for qubit in targets:
        acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[-1].id
        reference_value[qubit] = results[acq_handle].tolist()

    for twpa in twpas.values():
        twpa.connect()

    updates = []

    data = TwpaCalibrationData(
        twpa_power=twpa_power_ranges,
        twpa_frequency=twpa_frequency_ranges,
        reference_value=reference_value,
    )
    data_ = {qubit: [] for qubit in targets}
    for _pow in twpa_power_range:
        for twpa_channel in twpas:
            power = _pow + platform.config(twpa_channel).power
            updates.append({twpa_channel: {"power": power}})
        for freq in twpa_frequency_range:
            for twpa_channel in twpas:
                frequency = freq + platform.config(twpa_channel).frequency
                updates.append({twpa_channel: {"frequency": frequency}})
            results = platform.execute(
                [sequence],
                [sweepers],
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
                data_[qubit].append(results[acq_handle])
            for _ in twpas:
                updates.pop()
        for _ in twpas:
            updates.pop()
    data.data = {
        qubit: np.stack(data_[qubit], axis=0).reshape(
            len(twpa_power_range), len(twpa_frequency_range), len(frequency_range), 2
        )
        for qubit in targets
    }
    return data


def _fit(data: TwpaCalibrationData) -> TwpaCalibrationResults:
    """Post-processing function for TwpaCalibration.

    After computing the averaged gain we select the corresponding twpa frequency and power
    that maximizes the gain for each qubit.
    """
    gains = {}
    twpa_frequency = {}
    twpa_power = {}
    for qubit in data.qubits:
        averaged_gain = 20 * np.log10(
            np.mean(magnitude(data[qubit]), axis=2)
            / np.mean(magnitude(data.reference_value_array(qubit)), axis=0)
        )
        gains[qubit] = averaged_gain
        flat_index = np.argmax(averaged_gain)
        i, j = np.unravel_index(flat_index, averaged_gain.shape)
        twpa_frequency[qubit] = float(data.twpa_frequency[qubit][j])
        twpa_power[qubit] = float(data.twpa_power[qubit][i])
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
        ),
    )
    fig.update_xaxes(title_text="TWPA Frequency [GHz]")
    fig.update_yaxes(title_text="TWPA Power [dBm]")

    fig.update_layout(
        showlegend=False,
    )

    figures.append(fig)

    return figures, fitting_report


twpa_calibration = Routine(_acquisition, _fit, _plot)
"""Resonator TWPA Frequency Routine object."""
