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

    amplitude_min: float
    """Minimum flux pulse amplitude."""
    amplitude_max: float
    """Maximum flux pulse amplitude."""
    amplitude_step: float
    """Flux pulse amplitude step."""
    flux_pulse_amplitude: float
    """Flux pulse duration."""

    # duration_min: int
    # """Minimum flux pulse duration."""
    # duration_max: int
    # """Maximum flux duration start."""
    # duration_step: int
    # """Flux pulse duration step."""

    flux_pulse_duration: float
    """Flux pulse duration."""
    padding: int = 20
    """Time padding before and after flux pulse."""
    dt: int = 0
    """Time delay between flux pulse and basis rotation."""
    nshots: Optional[int] = None
    """Number of shots per point."""

    # flux_pulse_shapes
    # TODO support different shapes, for now only rectangular


@dataclass
class CryoscopeResults(Results):
    """Cryoscope outputs."""

    pass


# TODO: use probabilities
# CryoscopeType = np.dtype(
#    [("amp", np.float64), ("duration", np.float64), ("prob", np.float64)]
# )
CryoscopeType = np.dtype(
    [("amplitude", float), ("prob_0", np.float64), ("prob_1", np.float64)]
)
"""Custom dtype for Cryoscope."""


@dataclass
class CryoscopeData(Data):
    """Cryoscope acquisition outputs."""

    flux_pulse_duration: int
    data: dict[tuple[QubitId, str], npt.NDArray[CryoscopeType]] = field(
        default_factory=dict
    )

    def register_qubit(
        self,
        qubit: QubitId,
        tag: str,
        amps: npt.NDArray[np.int32],
        prob_0: npt.NDArray[np.float64],
        prob_1: npt.NDArray[np.float64],
    ):
        """Store output for a single qubit."""
        # size = len(amps) * len(durs)
        # amplitudes, durations = np.meshgrid(amps, durs)

        size = len(amps)
        # durations = amps

        ar = np.empty(size, dtype=CryoscopeType)
        ar["amplitude"] = amps.ravel()
        ar["prob_0"] = prob_0.ravel()
        ar["prob_1"] = prob_1.ravel()

        self.data[(qubit, tag)] = np.rec.array(ar)

    # def __getitem__(self, qubit):
    #     return {
    #         index: value
    #         for index, value in self.data.items()
    #         if set(qubit).issubset(index)
    #     }


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
            qubit, start=0, relative_phase=np.pi / 2
        )

        # TODO add support for flux pulse shapes
        # if params.flux_pulse_shapes and len(params.flux_pulse_shapes) == len(qubits):
        #     flux_pulse_shape = eval(params.flux_pulse_shapes[qubit])
        # else:
        #     flux_pulse_shape = Rectangular()
        flux_pulse_shape = Rectangular()
        flux_start = initial_pulses[qubit].finish + params.padding
        # apply a detuning flux pulse
        flux_pulses[qubit] = FluxPulse(
            start=flux_start,
            duration=params.flux_pulse_duration,
            amplitude=params.flux_pulse_amplitude,
            shape=flux_pulse_shape,
            channel=platform.qubits[qubit].flux.name,
            qubit=qubit,
        )

        # rotation_start = flux_start + params.duration_max + params.padding + params.dt
        # rotation_start = flux_start + params.duration_max + params.padding + params.dt
        # rotate around the X axis RX(-pi/2) to measure Y component
        rx90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=initial_pulses[qubit].finish
            + flux_pulses[qubit].finish
            + params.padding,
            # relative_phase=np.pi,
        )
        # rotate around the Y axis RX(-pi/2) to measure X component
        ry90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=initial_pulses[qubit].finish
            + flux_pulses[qubit].finish
            + params.padding,
            relative_phase=np.pi / 2,
        )

        # add readout at the end of the sequences
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

    amplitude_range = np.arange(
        params.amplitude_min, params.amplitude_max, params.amplitude_step
    )
    # duration_range = np.arange(
    #     params.duration_min, params.duration_max, params.duration_step
    # )

    amp_sweeper = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        pulses=list(flux_pulses.values()),
        type=SweeperType.FACTOR,
    )

    # dur_sweeper = Sweeper(
    #     Parameter.duration,
    #     duration_range,
    #     pulses=list(flux_pulses.values()),
    #     type=SweeperType.ABSOLUTE,
    # )

    options = ExecutionParameters(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = CryoscopeData(flux_pulse_duration=params.flux_pulse_duration)

    for sequence, tag in [(sequence_x, "MX"), (sequence_y, "MY")]:
        # results = platform.sweep(sequence, options, amp_sweeper, dur_sweeper)
        results = platform.sweep(sequence, options, amp_sweeper)
        for qubit in targets:
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(
                qubit,
                tag,
                amplitude_range * params.flux_pulse_amplitude,
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
        subplot_titles=(
            # f"Qubit {qubits[0]}",
            # f"Qubit {qubits[1]}",
        ),
    )
    qubit_X_data = data[(target, "MX")]
    qubit_Y_data = data[(target, "MY")]
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.amplitude,
            y=qubit_X_data.prob_1,
            name="X",
            legendgroup="X",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=qubit_Y_data.amplitude,
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

    phase = np.angle(X_exp + 1.0j * Y_exp)
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.amplitude,
            y=np.unwrap(phase),
            name="phase",
        ),
        row=2,
        col=1,
    )
    scipy.signal.savgol_filter(phase / 2 / np.pi, 13, 3, deriv=1, delta=0.001)
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.amplitude,
            y=np.unwrap(phase) / 2 / np.pi / data.flux_pulse_duration,
            name="Detuning [GHz]",
        ),
        row=3,
        col=1,
    )

    pol = np.polyfit(
        qubit_X_data.amplitude,
        np.unwrap(phase) / 2 / np.pi / data.flux_pulse_duration,
        deg=2,
    )
    print(pol)
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.amplitude,
            y=pol[2]
            + qubit_X_data.amplitude * pol[1]
            + qubit_X_data.amplitude**2 * pol[0],
            name="Fit Detuning [GHz]",
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        xaxis3_title="Flux pulse amplitude [a.u.]",
        yaxis1_title="Prob of 1",
        yaxis2_title="Phase [rad]",
        yaxis3_title="Detuning [GHz]",
    )
    return [fig], fitting_report


cryoscope_amplitude = Routine(_acquisition, _fit, _plot)
