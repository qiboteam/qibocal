from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.auto.history import History
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.cli.report import report
from qibocal.protocols.flux_dependence.utils import transmon_frequency

from . import utils

COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"

T1FluxType = np.dtype(
    [
        ("biases", np.float64),
        ("qubit_frequency", np.float64),
        ("T1", np.float64),
        ("T1_errors", np.float64),
    ]
)
"""Custom dtype for T1Flux routines."""


# TODO: Make this parameters a dict of 4 parameters classes for each routine ???
@dataclass
class T1FluxParameters(Parameters):
    """T1 runcard inputs."""

    biases_start: list[float]
    biases_end: list[float]
    biases_step: list[float]

    # Qubit spectroscopy
    freq_width: int
    """Width [Hz] for frequency sweep relative  to the qubit frequency."""
    freq_step: int
    """Frequency [Hz] step for sweep."""
    drive_duration: int
    """Drive pulse duration [ns]. Same for all qubits."""

    # Rabi amp signal
    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""

    # Flipping
    nflips_max: int
    """Maximum number of flips ([RX(pi) - RX(pi)] sequences). """
    nflips_step: int
    """Flip step."""

    # T1
    delay_before_readout_start: int
    """Initial delay before readout [ns]."""
    delay_before_readout_end: int
    """Final delay before readout [ns]."""
    delay_before_readout_step: int
    """Step delay before readout [ns]."""

    #  Ramsey signal
    detuning: int
    """Frequency detuning [Hz]."""
    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""

    # Optional qubit spectroscopy
    drive_amplitude: Optional[float] = None
    """Drive pulse amplitude (optional). Same for all qubits."""
    hardware_average: bool = True
    """By default hardware average will be performed."""
    # Optional rabi amp signal
    pulse_length: Optional[float] = None
    """RX pulse duration [ns]."""
    # Optional T1
    single_shot_T1: bool = False
    """If ``True`` save single shot  data."""


@dataclass
class T1FluxData(Data):
    """T1 acquisition outputs."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def average(self):
        if len(next(iter(self.data.values())).shape) > 1:
            return utils.average_single_shots(self.__class__, self.data)
        return self


@dataclass
class T1FluxResults(Results):
    sweetspot: dict[QubitId, float]
    "Best frequency for each qubit only regarding T1"


def _acquisition(
    params: T1FluxParameters, platform: Platform, targets: list[QubitId]
) -> T1FluxData:
    r"""Data acquisition for T1 experiment."""

    executor = Executor.create(name="myexec", platform=platform)
    from myexec import (
        close,
        flipping_signal,
        init,
        qubit_spectroscopy,
        rabi_amplitude_signal,
        ramsey_signal,
        single_shot_classification,
        t1,
    )

    data = T1FluxData()

    for target in targets:

        # TODO: Get the right parameters from the platform
        # params_qubit = {
        #     "w_max": platform.qubits[
        #         target
        #     ].drive_frequency,  # FIXME: this is not the qubit frequency
        #     "xj": 0,
        #     "d": platform.qubits[target].asymmetry,
        #     "normalization": platform.qubits[target].crosstalk_matrix[target],
        #     "offset": -platform.qubits[target].sweetspot
        #     * platform.qubits[target].crosstalk_matrix[
        #         target
        #     ],  # Check is this the right one ???
        #     "crosstalk_element": 1,
        #     "charging_energy": platform.qubits[target].Ec,
        # }

        # D2
        params_qubit = {
            "w_max": 5.552552628640306 * 1e9,  # FIXME: this is not the qubit frequency
            "xj": 0,
            "d": 0,
            "normalization": 0.8058267234810884,
            "offset": 0.0770175390610017,  # Check is this the right one ???
            "crosstalk_element": 1,
            "charging_energy": 0.2,
        }

        fit_function = transmon_frequency

        # TODO: Center around the sweetspot
        biases = np.arange(params.biases_start, params.biases_end, params.biases_step)

        i = 0
        for bias in biases:
            i += 1
            # Change the flux
            platform.qubits[target].flux.offset = bias

            # Change the qubit frequency
            qubit_frequency = fit_function(bias, **params_qubit)  # * 1e9
            platform.qubits[target].drive_frequency = qubit_frequency
            platform.qubits[target].native_gates.RX.frequency = qubit_frequency

            init(
                f"test_T1vsFlux/flux{i}", force=True, targets=targets
            )  # FIXME: Routine path

            # NOTE: Look and correct from the 1st estimate qubit frequency
            qubit_spectroscopy_output = qubit_spectroscopy(
                freq_width=params.freq_width,
                freq_step=params.freq_step,
                drive_duration=params.drive_duration,
                drive_amplitude=params.drive_amplitude,
                relaxation_time=5000,
                nshots=1024,
            )

            qubit_spectroscopy_output.update_platform(platform)

            # Set maximun drive amplitude
            platform.qubits[target].native_gates.RX.amplitude = (
                0.5  # FIXME: For QM this should be 0.5
            )
            platform.qubits[target].native_gates.RX.duration = params.pulse_length
            if qubit_spectroscopy_output.results.frequency:
                platform.qubits[target].native_gates.RX.frequency = (
                    qubit_spectroscopy_output.results.frequency[target]
                )
            else:
                platform.qubits[target].native_gates.RX.frequency = qubit_frequency

            rabi_output = rabi_amplitude_signal(
                min_amp_factor=params.min_amp_factor,
                max_amp_factor=params.max_amp_factor,
                step_amp_factor=params.step_amp_factor,
                pulse_length=platform.qubits[target].native_gates.RX.duration,
            )

            if rabi_output.results.amplitude[target] > 0.5:
                print(
                    f"Rabi fit has pi pulse amplitude {rabi_output.results.amplitude[target]}, greater than 0.5 not possible for QM. Skipping to next bias point."
                )
                continue
            rabi_output.update_platform(platform)

            ramsey_output = ramsey_signal(
                delay_between_pulses_start=params.delay_between_pulses_start,
                delay_between_pulses_end=params.delay_between_pulses_end,
                delay_between_pulses_step=params.delay_between_pulses_step,
                detuning=params.detuning,
            )
            ramsey_output.update_platform(platform)

            flipping_output = flipping_signal(
                nflips_max=params.nflips_max,
                nflips_step=params.nflips_step,
            )
            flipping_output.update_platform(platform)

            discrimination_output = single_shot_classification(nshots=15000)
            discrimination_output.update_platform(platform)

            t1_output = t1(
                delay_before_readout_start=params.delay_before_readout_start,
                delay_before_readout_end=params.delay_before_readout_end,
                delay_before_readout_step=params.delay_before_readout_step,
                single_shot=params.single_shot_T1,
            )

            data.register_qubit(
                T1FluxType,
                (target),
                dict(
                    biases=[bias],
                    qubit_frequency=[platform.qubits[target].native_gates.RX.frequency],
                    T1=[t1_output.results.t1[target][0]],
                    T1_errors=[t1_output.results.t1[target][1]],
                ),
            )

            # close()
            report(executor.path, executor.history)
            executor.history = History()

    # stop and disconnect platform
    close()

    return data


def _fit(data: T1FluxData) -> T1FluxResults:
    """Post-processing function for 1Flux."""
    # TODO: Highest T1+T2 at the lowest frequency cost function

    return T1FluxResults(sweetspot=None)


def _plot(data: T1FluxData, target: QubitId, fit=None):
    """Plotting function for T1 experiment."""

    figures = []

    qubit_data = data[target]
    qubit_frequency = qubit_data.qubit_frequency
    t1 = qubit_data.T1
    error_bars = qubit_data.T1_errors

    fig = go.Figure(
        [
            go.Scatter(
                x=qubit_frequency,
                y=t1,
                opacity=1,
                name="Probability of 1",
                showlegend=True,
                legendgroup="Probability of 1",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((qubit_frequency, qubit_frequency[::-1])),
                y=np.concatenate((t1 + error_bars, (t1 - error_bars)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    # last part
    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHZ]",
        yaxis_title="T1 [ns]",
    )

    figures.append(fig)

    return figures, ""


def _update(results, platform: Platform, target: QubitId):
    pass


t1_flux = Routine(_acquisition, _fit, _plot, _update)
"""T1 Flux Routine object."""
