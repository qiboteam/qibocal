"""Protocol to calibrate TWPA power and frequency for a specific probe frequency."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Platform

from ...auto.operation import Data, Parameters, QubitId, Results, Routine
from ...result import magnitude
from ..utils import HZ_TO_GHZ


@dataclass
class TwpaCalibrationParameters(Parameters):
    """TwpaCalibration runcard inputs."""

    probe_frequency: float

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

    twpa_frequency: dict[QubitId, float] = field(default_factory=dict)
    """TWPA frequency [GHz] for each qubit."""


TwpaCalibrationType = np.dtype(
    [
        ("twpa_power", np.float64),
        ("twpa_freq", np.float64),
        ("signal", np.float64),
    ]
)
"""Custom dtype for TwpaCalibration."""


@dataclass
class TwpaCalibrationData(Data):
    """TwpaCalibration data acquisition."""

    data: dict[QubitId, npt.NDArray[TwpaCalibrationType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: TwpaCalibrationParameters,
    platform: Platform,
    targets: list[QubitId],
) -> TwpaCalibrationData:
    data = TwpaCalibrationData()

    assert len(targets) == 1, "Twpa calibration can be executed on one qubit at a time."

    frequency_range = np.arange(
        -params.twpa_freq_width / 2, params.twpa_freq_width / 2, params.twpa_freq_step
    )

    power_range = np.arange(
        -params.twpa_pow_width / 2, params.twpa_pow_width / 2, params.twpa_pow_step
    )
    qubit = targets[0]
    sequence = platform.natives.single_qubit[qubit].MZ()
    ro_probe = platform.qubits[qubit].probe
    twpa = platform.channels[ro_probe].lo
    updates = []
    updates.append({ro_probe: {"frequency": params.probe_frequency}})
    for _pow in power_range:
        power = _pow + platform.config(twpa).power
        updates.append({twpa: {"power": power}})
        for freq in frequency_range:
            frequency = freq + platform.config(twpa).frequency
            updates.append({twpa: {"frequency": frequency}})

            results = platform.execute(
                [sequence],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
                updates=updates,
            )
            acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[
                -1
            ].id
            data.register_qubit(
                TwpaCalibrationType,
                qubit,
                dict(
                    twpa_freq=np.array([frequency]),
                    twpa_power=np.array([power]),
                    signal=np.array(magnitude([results[acq_handle]])),
                ),
            )
            updates.pop()
        updates.pop()

    return data


def _fit(data: TwpaCalibrationData) -> TwpaCalibrationResults:
    """Post-processing function for TwpaCalibration."""
    return TwpaCalibrationResults()


def _plot(data: TwpaCalibrationData, fit: TwpaCalibrationResults, target):
    """Plotting for TwpaCalibration."""

    figures = []
    fitting_report = ""
    fig = go.Figure()

    fitting_report = ""
    qubit_data = data[target]
    twpa_power = qubit_data.twpa_power
    twpa_frequencies = qubit_data.twpa_freq * HZ_TO_GHZ

    fig.add_trace(
        go.Heatmap(
            x=twpa_frequencies,
            y=twpa_power,
            z=qubit_data.signal,
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
