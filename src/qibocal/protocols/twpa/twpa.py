"""Protocol to calibrate TWPA power and frequency for a specific probe frequency."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Platform, Sweeper

from ...auto.operation import Data, Parameters, QubitId, Results, Routine
from ...result import magnitude
from ..utils import HZ_TO_GHZ, table_dict, table_html


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
    twpa_frequency: float
    """TWPA frequency [GHz]."""
    twpa_power: float
    """TWPA power [dBm]."""


@dataclass
class TwpaCalibrationData(Data):
    """TwpaCalibration data acquisition."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""
    twpa_frequency: list[float] = field(default=list)
    """List with twpa frequency values swept."""
    twpa_power: list[float] = field(default=list)
    """List with twpa power values swept."""
    reference_value: list[float] = field(default=list)
    """Values for readout frequency sweep with TWPA off."""

    @property
    def reference_value_array(self) -> npt.NDArray:
        """Return reference value as a numpy array."""
        return np.array(self.reference_value).reshape(-1, 2)


def _acquisition(
    params: TwpaCalibrationParameters,
    platform: Platform,
    targets: list[QubitId],
) -> TwpaCalibrationData:
    """Acquisition function for TwpaCalibration.

    This protocol assumes that only target is provided and it will fail otherwise.
    First we acquire the reference value without TWPA, then we sweep the TWPA power and frequency.
    The gain is computed as the norm of the complex readout signal minus the norm of the complex readout signal without TWPA.
    """
    assert len(targets) == 1, "Twpa calibration can be executed on one qubit at a time."

    qubit = targets[0]
    sequence = platform.natives.single_qubit[qubit].MZ()
    acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[-1].id
    ro_probe = platform.qubits[qubit].probe
    twpa_channel = platform.channels[platform.qubits[qubit].acquisition].twpa_pump
    frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    twpa = platform.instruments[
        platform.channels[platform.qubits[qubit].acquisition].twpa_pump
    ]
    twpa_frequency_range = (
        np.arange(
            -params.twpa_freq_width / 2,
            params.twpa_freq_width / 2,
            params.twpa_freq_step,
        )
        + platform.config(
            platform.channels[platform.qubits[qubit].acquisition].twpa_pump
        ).frequency
    )
    twpa_power_range = (
        np.arange(
            -params.twpa_pow_width / 2, params.twpa_pow_width / 2, params.twpa_pow_step
        )
        + platform.config(
            platform.channels[platform.qubits[qubit].acquisition].twpa_pump
        ).power
    )

    updates = []
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=platform.config(ro_probe).frequency + frequency_range,
        channels=[ro_probe],
    )

    # reference value without twpa
    # TODO: check if this will work on hardware
    twpa.disconnect()
    reference_value = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
        updates=updates,
    )[acq_handle]
    twpa.connect()

    data = TwpaCalibrationData(
        twpa_power=twpa_power_range.tolist(),
        twpa_frequency=twpa_frequency_range.tolist(),
        reference_value=reference_value.tolist(),
    )
    _data = []
    for _pow in twpa_power_range:
        power = _pow + platform.config(twpa_channel).power
        updates.append({twpa_channel: {"power": power}})
        for freq in twpa_frequency_range:
            frequency = freq + platform.config(twpa_channel).frequency
            updates.append({twpa_channel: {"frequency": frequency}})
            results = platform.execute(
                [sequence],
                [[sweeper]],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
                updates=updates,
            )
            acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[
                -1
            ].id
            _data.append(results[acq_handle])
            updates.pop()
        updates.pop()
    data.data = {
        qubit: np.stack(_data, axis=0).reshape(
            len(twpa_power_range), len(twpa_frequency_range), len(frequency_range), 2
        )
    }
    return data


def _fit(data: TwpaCalibrationData) -> TwpaCalibrationResults:
    """Post-processing function for TwpaCalibration.

    After computing the averaged gain we select the corresponding twpa frequency and power
    that maximizes the gain.
    """
    qubit = data.qubits[0]
    averaged_gain = np.mean(magnitude(data[qubit]), axis=2) / np.mean(
        magnitude(data.reference_value_array), axis=0
    )
    flat_index = np.argmax(np.abs(averaged_gain))
    i, j = np.unravel_index(flat_index, averaged_gain.shape)
    twpa_frequency = data.twpa_frequency[j]
    twpa_power = data.twpa_power[i]
    return TwpaCalibrationResults(
        data={qubit: averaged_gain},
        twpa_frequency=float(twpa_frequency),
        twpa_power=float(twpa_power),
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
                    np.round(fit.twpa_frequency, 4),
                    np.round(fit.twpa_power, 4),
                ],
            )
        )
        averaged_gain = fit.data[target]
    else:
        averaged_gain = np.mean(magnitude(data[target]), axis=2) / np.mean(
            magnitude(data.reference_value_array), axis=0
        )
        fitting_report = ""

    fig.add_trace(
        go.Heatmap(
            x=np.array(data.twpa_frequency) * HZ_TO_GHZ,
            y=data.twpa_power,
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
