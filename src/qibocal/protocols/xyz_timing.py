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

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.calibration.calibration import QubitId
from qibocal.calibration.platform import CalibrationPlatform
from qibocal.result import probability

COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"


@dataclass
class XYZTimingResults(Results):
    pass


@dataclass
class XYZTimingParameters(Parameters):

    flux_amplitude: float
    delay_step: float
    delay_stop: float
    flux_pulse_duration: float


XYZTimingType = np.dtype(
    [("delay", np.float64), ("prob", np.float64), ("errors", np.float64)]
)


@dataclass
class XYZTimingData(Data):

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: XYZTimingParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> XYZTimingData:

    data = XYZTimingData()
    natives = platform.natives.single_qubit
    delays = np.arange(
        0,
        params.delay_stop,
        params.delay_step,
    )
    sequence = PulseSequence()
    flux_delays = []
    for qubit in targets:
        drive_channel = platform.qubits[qubit].drive
        flux_channel = platform.qubits[qubit].flux
        ro_channel = platform.qubits[qubit].acquisition
        drive_pulse = natives[qubit].RX()[0]
        readout_pulse = natives[qubit].MZ()[0]

        flux_pulse = Pulse(
            duration=params.flux_pulse_duration,
            amplitude=params.flux_amplitude,
            relative_phase=0,
            envelope=Rectangular(),
        )
        qd_delay = Delay(duration=params.flux_pulse_duration)
        flux_delay = Delay(duration=0)
        flux_delays.append(flux_delay)

        sequence.extend(
            [
                (drive_channel, qd_delay),
                drive_pulse,
                (flux_channel, flux_delay),
                (flux_channel, flux_pulse),
            ]
        )
        sequence.align([drive_channel, flux_channel, ro_channel])
        sequence.append(readout_pulse)

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=delays,
        pulses=flux_delays,
    )

    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        probs = probability(results[ro_pulse.id], state=1)
        # The probability errors are the standard errors of the binomial distribution
        errors = [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs]
        data.register_qubit(
            XYZTimingType,
            (qubit),
            dict(
                delay=delays,
                prob=probs,
                errors=errors,
            ),
        )
    return data


def _fit(data: XYZTimingData) -> XYZTimingResults:
    return XYZTimingResults()


def _plot(data: XYZTimingData, target: QubitId, fit: XYZTimingResults = None):
    figures = []
    qubit_data = data.data[target]
    delays = qubit_data.delay
    probs = qubit_data.prob
    error_bars = qubit_data.errors
    fig = go.Figure(
        [
            go.Scatter(
                x=delays,
                y=probs,
                opacity=1,
                name="Probability of State 0",
                showlegend=True,
                legendgroup="Probability of State 0",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((delays, delays[::-1])),
                y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )
    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Ground state probability",
    )

    figures.append(fig)

    return figures, ""


xyz_timing = Routine(_acquisition, _fit, _plot)
