from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import probability

from ..rabi.utils import fit_length_function, rabi_length_function
from ..utils import (
    COLORBAND,
    COLORBAND_LINE,
    fallback_period,
    guess_period,
    table_dict,
    table_html,
)

__all__ = ["coupler_readout"]


@dataclass
class CouplerReadoutParameters(Parameters):
    """Coupler readout runcard inputs."""

    delay_min: float
    delay_max: float
    delay_step: float
    drive_duration: float
    drive_amplitude: float


@dataclass
class CouplerReadoutResults(Results):
    """Coupler readout fit."""

    readout_duration: dict[QubitPairId, float] = field(default_factory=dict)
    fitted_parameters: dict[QubitPairId, list[float]] = field(default_factory=dict)


@dataclass
class CouplerReadoutData(Data):
    """Coupler acquisition outputs."""

    waits: list = field(default_factory=list)
    data: dict[QubitPairId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    def probability(self, pair: QubitPairId) -> npt.NDArray:
        """Return the probability data for a specific qubit."""
        return probability(self.data[pair], state=1)

    def error(self, pair: QubitPairId) -> npt.NDArray:
        """Return the error data for a specific qubit."""
        probs = self.probability(pair)
        nshots = self.data[pair].shape[0]
        return np.sqrt(probs * (1 - probs) / nshots)


def _acquisition(
    params: CouplerReadoutParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerReadoutData:
    """Readout sequence for coupler taken from https://arxiv.org/pdf/2503.13225."""

    assert len(targets) == 1, (
        "Coupler readout is only available for one qubit pair at a time."
    )
    pair = targets[0]
    waits = np.arange(
        params.delay_min,
        params.delay_max,
        params.delay_step,
    )

    data = CouplerReadoutData()

    sequence = PulseSequence()
    delays = 2 * [Delay(duration=0)]
    ry90_sequence = platform.natives.single_qubit[pair[1]].R(
        theta=np.pi / 2, phi=np.pi / 2
    )
    sequence.append(
        (
            platform.qubits[pair[0]].drive,
            Pulse(
                duration=params.drive_duration,
                amplitude=params.drive_amplitude,
                envelope=Rectangular(),
            ),
        )
    )
    sequence.append(
        (platform.qubits[pair[1]].drive, Delay(duration=params.drive_duration))
    )
    sequence.append(
        (platform.qubits[pair[1]].acquisition, Delay(duration=params.drive_duration))
    )
    sequence += ry90_sequence
    sequence.append(
        (platform.qubits[pair[1]].acquisition, Delay(duration=ry90_sequence.duration))
    )
    sequence.append((platform.qubits[pair[1]].drive, delays[0]))
    sequence.append((platform.qubits[pair[1]].acquisition, delays[1]))
    sequence += platform.natives.single_qubit[pair[1]].R(theta=np.pi / 2, phi=np.pi / 2)
    sequence.append(
        (platform.qubits[pair[1]].acquisition, Delay(duration=ry90_sequence.duration))
    )
    sequence += platform.natives.single_qubit[pair[1]].MZ()

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=waits,
        pulses=delays,
    )

    data = CouplerReadoutData(waits=waits.tolist())

    updates = [
        {
            platform.qubits[pair[0]].drive: {
                "frequency": platform.calibration.couplers[
                    platform.calibration.two_qubits[pair].coupler
                ].qubit.frequency_01
            }
        },
        {
            platform.couplers[platform.calibration.two_qubits[pair].coupler].flux: {
                "offset": platform.calibration.couplers[
                    platform.calibration.two_qubits[pair].coupler
                ].qubit.sweetspot
            }
        },
    ]

    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
        updates=updates,
    )
    handle = list(sequence.channel(platform.qubits[pair[1]].acquisition))[-1].id
    data.data[pair] = results[handle]

    return data


def _fit(data: CouplerReadoutData) -> CouplerReadoutResults:
    """ ""Fitting routine for Coupler readout experiment."""
    fitted_parameters = {}
    readout_duration = {}

    for pair in data.pairs:
        raw_x = np.array(data.waits)
        min_x = np.min(raw_x)
        max_x = np.max(raw_x)
        y = data.probability(pair)
        x = (raw_x - min_x) / (max_x - min_x)
        period = fallback_period(guess_period(x, y))
        pguess = [0.5, 0.5, period, 0, 0]

        try:
            popt, perr, pi_pulse_parameter = fit_length_function(
                x,
                y,
                pguess,
                sigma=data.error(pair),
                signal=False,
                x_limits=(min_x, max_x),
            )
            readout_duration[pair] = [pi_pulse_parameter, perr[2] * (max_x - min_x) / 2]
            fitted_parameters[pair] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for coupler of pair {pair} due to {e}.")

    return CouplerReadoutResults(
        readout_duration=readout_duration, fitted_parameters=fitted_parameters
    )


def _plot(
    data: CouplerReadoutData, target: QubitPairId, fit: CouplerReadoutResults = None
):
    """Plotting function for CouplerReadout Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""
    waits = data.waits
    prob = data.probability(target)
    error = data.error(target)
    fig.add_traces(
        [
            go.Scatter(
                x=waits,
                y=prob,
                opacity=1,
                name="Probability",
                showlegend=True,
                legendgroup="Probability",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((prob, prob[::-1])),
                y=np.concatenate((prob + error, (prob - error)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    if fit is not None:
        rabi_parameter_range = np.linspace(
            min(waits),
            max(waits),
            2 * len(waits),
        )
        params = fit.fitted_parameters[target]
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=rabi_length_function(rabi_parameter_range, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
        )

        fitting_report = table_html(
            table_dict(
                [target],
                [
                    "Pulse length [ns]",
                ],
                [fit.readout_duration[target]],
                display_error=True,
            )
        )
    fig.update_layout(
        showlegend=True,
        xaxis_title="Duration [ns]",
        yaxis_title="Probability of |1>",
    )
    figures.append(fig)

    return figures, fitting_report


coupler_readout = Routine(_acquisition, _fit, _plot)
"""CouplerReadout Routine object."""
