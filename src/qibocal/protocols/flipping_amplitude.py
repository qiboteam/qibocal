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
    return None


def _fit(data: FlippingAmplitudeData) -> FlippingAmplitudeResults:
    return None


def _plot(
    data: FlippingAmplitudeData,
    target: QubitId,
    fit: FlippingAmplitudeResults | None = None,
):
    return None


def _update(
    results: FlippingAmplitudeResults,
    platform: CalibrationPlatform,
    qubit: QubitId,
):
    update.drive_amplitude(results.amplitude[qubit], results.rx90, platform, qubit)


flipping_amplitude = Routine(_acquisition, _fit, _plot, _update)
"""FlippingAmplitude Routine object."""
