"""CZ virtual correction experiment for two qubit gates, tune landscape."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitPairId,
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform

from .utils import order_pair

__all__ = ["phase_calibration"]


# TODO: add option to pass virtual Z phase and duration
@dataclass
class PhaseCalibrationParameters(Parameters):
    """VirtualZ runcard inputs."""

    amplitude_min: float
    amplitude_max: float
    amplitude_step: float
    native: str = "CZ"
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """

    @property
    def amplitude_range(self):
        return np.arange(
            self.amplitude_min, self.amplitude_max, self.amplitude_step
        ).tolist()


@dataclass
class PhaseCalibrationResults(Results):
    """VirtualZ outputs when fitting will be done."""


@dataclass
class PhaseCalibrationData(Data):
    """PhaseCalibration data."""

    native: str
    amplitude_range: list[float] = field(default_factory=list)
    data: dict[QubitPairId, npt.NDArray] = field(default_factory=dict)


def _acquisition(
    params: PhaseCalibrationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> PhaseCalibrationData:
    r"""
    Acquisition for PhaseCalibration.

    Check the two-qubit landscape created by a flux pulse of a given duration
    and amplitude.
    The system is initialized with a X90 pulse on the low frequency qubit and either
    an Id or an X gate on the high frequency qubit. Then the flux pulse is applied to
    the high frequency qubit in order to perform a two-qubit interaction.
    A $X_{\beta}90$ pulse is applied to the low frequency qubit before measurement.
    That is, a pi-half pulse around the relative phase parametereized by the angle theta.
    Measurements on the low frequency qubit yield the 2Q-phase of the gate and the
    remnant single qubit Z phase aquired during the execution to be corrected.
    Population of the high frequency qubit yield the leakage to the non-computational states
    during the execution of the flux pulse.
    """
    data = PhaseCalibrationData(
        native=params.native,
        amplitude_range=params.amplitude_range,
    )
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ordered_pair = order_pair(pair, platform)
        low_natives = platform.natives.single_qubit[ordered_pair[0]]
        high_natives = platform.natives.single_qubit[ordered_pair[1]]
        sequence = PulseSequence()
        sequence += low_natives.R(theta=np.pi / 2)
        sequence += high_natives.RX()
        sequence |= getattr(platform.natives.two_qubit[ordered_pair], params.native)()
        sequence |= low_natives.R(theta=np.pi / 2)
        sequence |= low_natives.MZ() + high_natives.MZ()

        sweeper = Sweeper(
            parameter=Parameter.amplitude,
            range=(
                params.amplitude_min,
                params.amplitude_max,
                params.amplitude_step,
            ),
            pulses=list(sequence.channel(platform.qubits[ordered_pair[1]].flux)),
        )
        results = platform.execute(
            [sequence],
            [[sweeper]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        ro_low = list(sequence.channel(platform.qubits[ordered_pair[0]].acquisition))[
            -1
        ]
        ro_high = list(sequence.channel(platform.qubits[ordered_pair[1]].acquisition))[
            -1
        ]

        data.data[pair] = np.stack([results[ro_low.id], results[ro_high.id]])
    return data


def _fit(
    data: PhaseCalibrationData,
) -> PhaseCalibrationResults:
    r"""Fitting routine for the experiment.

    The used model is

    .. math::

        y = p_0 sin\Big(x + p_2\Big) + p_1.
    """
    return PhaseCalibrationResults()


def _plot(
    data: PhaseCalibrationData, fit: PhaseCalibrationResults, target: QubitPairId
):
    """Plot routine for PhaseCalibration."""
    if target not in data.data:
        target = (target[1], target[0])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {target[0]} - Low Frequency",
            f"Qubit {target[1]} - High Frequency",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.amplitude_range,
            y=data.data[target][0],
            mode="markers",
        ),
        col=1,
        row=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.amplitude_range,
            y=data.data[target][1],
            mode="markers",
        ),
        col=2,
        row=1,
    )
    fitting_report = ""

    fig.update_xaxes(title_text="Qubit flux")
    fig.update_yaxes(title_text="Probability")

    return [fig], fitting_report


def _update(
    results: PhaseCalibrationResults, platform: CalibrationPlatform, target: QubitPairId
):
    pass


phase_calibration = Routine(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Virtual phases correction protocol."""
