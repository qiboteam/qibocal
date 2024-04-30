"""Cryoscope experiment, corrects distortions."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy
from plotly.subplots import make_subplots
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.platform import Platform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Results, Routine


@dataclass
class CryoscopeParameters(Parameters):
    """Cryoscope runcard inputs."""

    duration_min: int
    """Minimum flux pulse duration."""
    duration_max: int
    """Maximum flux duration start."""
    duration_step: int
    """Flux pulse duration step."""
    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    padding: int = 20
    """Time padding before and after flux pulse."""
    nshots: Optional[int] = None
    """Number of shots per point."""

    # flux_pulse_shapes
    # TODO support different shapes, for now only rectangular


@dataclass
class CryoscopeResults(Results):
    """Cryoscope outputs."""

    pass


CryoscopeType = np.dtype(
    [("duration", int), ("prob_0", np.float64), ("prob_1", np.float64)]
)
"""Custom dtype for Cryoscope."""


@dataclass
class CryoscopeData(Data):
    """Cryoscope acquisition outputs."""

    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    data: dict[tuple[QubitId, str], npt.NDArray[CryoscopeType]] = field(
        default_factory=dict
    )

    def register_qubit(
        self,
        qubit: QubitId,
        tag: str,
        durs: npt.NDArray[np.int32],
        prob_0: npt.NDArray[np.float64],
        prob_1: npt.NDArray[np.float64],
    ):
        """Store output for a single qubit."""

        size = len(durs)
        durations = durs

        ar = np.empty(size, dtype=CryoscopeType)
        ar["duration"] = durations.ravel()
        ar["prob_0"] = prob_0.ravel()
        ar["prob_1"] = prob_1.ravel()

        self.data[(qubit, tag)] = np.rec.array(ar)


def _acquisition(
    params: CryoscopeParameters,
    platform: Platform,
    targets: list[QubitId],
) -> CryoscopeData:
    # define sequences of pulses to be executed
    sequence_x = PulseSequence()
    sequence_y = PulseSequence()

    initial_pulses = {}
    flux_pulses = {}
    rx90_pulses = {}
    ry90_pulses = {}
    ro_pulses = {}

    for qubit in targets:

        initial_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=0,
            relative_phase=np.pi / 2,
        )

        # TODO add support for flux pulse shapes
        flux_pulse_shape = Rectangular()
        flux_start = initial_pulses[qubit].finish + params.padding
        # apply a detuning flux pulse
        flux_pulses[qubit] = FluxPulse(
            start=flux_start,
            duration=params.duration_min,
            amplitude=params.flux_pulse_amplitude,
            shape=flux_pulse_shape,
            channel=platform.qubits[qubit].flux.name,
            qubit=qubit,
        )

        rx90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=params.duration_max + params.padding,
        )

        ry90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=params.duration_max + params.padding,
            relative_phase=np.pi / 2,
        )

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=rx90_pulses[qubit].finish  # to be fixed
        )

        # create the sequences
        sequence_x.add(
            initial_pulses[qubit],
            flux_pulses[qubit],
            ry90_pulses[qubit],  # rotate around Y to measure X CHECK
            ro_pulses[qubit],
        )
        sequence_y.add(
            initial_pulses[qubit],
            flux_pulses[qubit],
            rx90_pulses[qubit],  # rotate around X to measure Y CHECK
            ro_pulses[qubit],
        )

    duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    dur_sweeper = Sweeper(
        Parameter.duration,
        duration_range,
        pulses=list(flux_pulses.values()),
        type=SweeperType.ABSOLUTE,
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = CryoscopeData(flux_pulse_amplitude=params.flux_pulse_amplitude)
    for sequence, tag in [(sequence_x, "MX"), (sequence_y, "MY")]:
        results = platform.sweep(sequence, options, dur_sweeper)
        for qubit in targets:
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(
                qubit,
                tag,
                duration_range,
                result.probability(state=0),
                result.probability(state=1),
            )

    return data


def _fit(data: CryoscopeData) -> CryoscopeResults:
    return CryoscopeResults()


def _plot(data: CryoscopeData, fit: CryoscopeResults, target: QubitId):
    """Cryoscope plots."""
    figures = []

    fitting_report = f"Cryoscope of qubit {target}"

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(),
    )
    qubit_X_data = data[(target, "MX")]
    qubit_Y_data = data[(target, "MY")]
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=qubit_X_data.prob_1,
            name="X",
            legendgroup="X",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=qubit_Y_data.duration,
            y=qubit_Y_data.prob_1,
            name="Y",
            legendgroup="Y",
        ),
        row=1,
        col=1,
    )

    # minus sign for X_exp becuase I get -cos phase
    X_exp = qubit_X_data.prob_1 - qubit_X_data.prob_0
    Y_exp = qubit_Y_data.prob_0 - qubit_Y_data.prob_1
    phase = np.unwrap(np.angle(X_exp + 1.0j * Y_exp))
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=phase,
            name="phase",
        ),
        row=2,
        col=1,
    )

    coeffs = [-9.94466439e00, -5.88747144e-02, 2.31272909e-05]
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=scipy.signal.savgol_filter(
                (phase - phase[-1]) / 2 / np.pi,
                13,
                3,
                deriv=1,
            ),
            name="detuning",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=np.polyval(
                coeffs,
                (data.flux_pulse_amplitude) * np.ones(len(qubit_X_data.duration)),
            ),
            name="fit",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        xaxis3_title="Flux pulse duration [ns]",
        yaxis1_title="Prob of 1",
        yaxis2_title="Phase [rad]",
        yaxis3_title="Detuning [GHz]",
    )
    return [fig], fitting_report


cryoscope = Routine(_acquisition, _fit, _plot)
