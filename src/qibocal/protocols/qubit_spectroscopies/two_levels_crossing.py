from dataclasses import dataclass

import numpy as np
from plotly.subplots import go, make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    ParallelSweepers,
    Parameter,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    Sweeper,
)
from scipy.constants import nano

from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import Range, RangeLike, table_dict, table_html, to_range
from qibocal.result import magnitude, phase


@dataclass
class TwoLevelsCrossingParameters(Parameters):
    drive1: RangeLike
    """First tone frequency range for sweep [Hz]."""
    drive2: RangeLike
    """Second tone frequency range for sweep [Hz]."""
    duration: float = 4000
    """Spectroscopic pulses duration."""
    amplitude: float | tuple[float, float] = 1.0
    """Spectroscopic pulses amplitude."""


TwoLevelsType = np.dtype([("i", np.float64), ("q", np.float64)])
"""Two levels crossing spectroscopy results quadratures."""


@dataclass
class TwoLevelsCrossingData(Data):
    """Data."""

    drive1: dict[QubitId, Range]
    drive2: dict[QubitId, Range]
    data: dict[QubitId, np.ndarray]
    """Raw data acquired, IQ components of the readout signal."""

    def signal(self, qubit: QubitId) -> np.ndarray:
        return magnitude(self.data[qubit])

    def phase(self, qubit: QubitId) -> np.ndarray:
        return phase(self.data[qubit])

    def grid(self, qubit: QubitId) -> tuple[np.ndarray, np.ndarray]:
        drive1, drive2 = self.coords(qubit)
        return np.meshgrid(drive1, drive2)

    def coords(self, qubit: QubitId) -> tuple[np.ndarray, np.ndarray]:
        return np.arange(*self.drive1[qubit]), np.arange(*self.drive2[qubit])


@dataclass
class TwoLevelsCrossingResults(Results):
    """Fits outputs."""


def _acquisition(
    params: TwoLevelsCrossingParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> TwoLevelsCrossingData:
    drive1: dict[QubitId, Range] = {}
    drive2: dict[QubitId, Range] = {}
    if isinstance(params.amplitude, tuple):
        amp1, amp2 = params.amplitude
    else:
        amp1 = amp2 = params.amplitude

    sequence = PulseSequence()
    sweep1: ParallelSweepers = []
    sweep2: ParallelSweepers = []
    readouts: dict[QubitId, Readout] = {}
    for q in targets:
        qubit = platform.qubits[q]
        drive12 = qubit.drive_extra[(1, 2)]
        natives = platform.natives.single_qubit[q]
        assert qubit.drive is not None
        assert drive12 is not None
        assert natives.MZ is not None
        readout = natives.MZ()
        assert isinstance(readout[0][1], Readout)
        readouts[q] = readout[0][1]

        tone1 = Pulse(amplitude=amp1, duration=params.duration, envelope=Rectangular())
        tone2 = tone1.new().model_copy(update={"amplitude": amp2})
        sequence += [
            (qubit.drive, tone1),
            (drive12, tone2),
            (qubit.acquisition, Delay(duration=params.duration)),
        ]
        sequence += readout

        frequency = platform.calibration.single_qubits[q].qubit.frequency_01
        drive1[q] = to_range(params.drive1, center=frequency)
        drive2[q] = to_range(params.drive2, center=frequency)
        sweep1.append(
            Sweeper(
                parameter=Parameter.frequency, range=drive1[q], channels=[qubit.drive]
            )
        )
        sweep2.append(
            Sweeper(parameter=Parameter.frequency, range=drive2[q], channels=[drive12])
        )

    results = platform.execute(
        [sequence],
        [sweep2, sweep1],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    return TwoLevelsCrossingData(
        drive1=drive1,
        drive2=drive2,
        data={q: results[readouts[q].id] for q in targets},
    )


def _fit(
    data: TwoLevelsCrossingData,
) -> TwoLevelsCrossingResults:
    # ignored
    _ = data
    return TwoLevelsCrossingResults()


def _plot(data: TwoLevelsCrossingData, target: QubitId, fit: TwoLevelsCrossingResults):
    x, y = tuple(c * nano for c in data.coords(target))
    signal = data.signal(target)
    phase = data.phase(target)

    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Raw Signal [a.u.]",
            "Phase [rad]",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=signal,
            zmin=np.percentile(signal, 0.5),
            zmax=np.percentile(signal, 99.5),
            colorbar=dict(title="Raw signal"),
            colorbar_x=1.01,
            colorscale="Viridis",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=phase,
            colorbar_x=0.46,
            colorscale="Viridis",
        ),
        row=1,
        col=2,
    )

    if fit is not None:
        labels = values = []
        fitting_report = table_html(table_dict(target, labels, values))

    fig.update_layout(showlegend=True, legend=dict(orientation="h"))
    fig.update_xaxes(title_text="First tone [GHz]")
    fig.update_yaxes(title_text="Second tone [GHz]")
    figures.append(fig)

    return figures, fitting_report


def _update(
    results: TwoLevelsCrossingResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    # ignored
    _ = results, platform, target


two_levels_crossing = Protocol(_acquisition, _fit, _plot, _update)
"""Two levels crossing spectroscopy protocol."""
