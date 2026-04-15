from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    ParallelSweepers,
    Parameter,
    Pulse,
    PulseSequence,
    Sweeper,
)
from qibolab._core.pulses import Rectangular
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.two_qubit_interaction.chevron.utils import COLORAXIS
from qibocal.protocols.two_qubit_interaction.utils import order_pair
from qibocal.protocols.utils import HZ_TO_MHZ, table_dict, table_html

from .utils import chevron_function

__all__ = ["coupler_chevron_time"]


class InvalidInputParameters(Exception):
    pass


@dataclass
class CouplerChevronTimeParameters(Parameters):
    """CouplerChevronTime runcard inputs."""

    coupler_ampl_range: tuple[float, float, float] | None = None
    """Coupler amplitude range."""
    coupler_ampl_min: float | None = None
    """Coupler amplitude minimum."""
    coupler_ampl_max: float | None = None
    """Coupler amplitude maximum."""
    coupler_ampl_step: float | None = None
    """Coupler amplitude step."""
    duration_range: tuple[int, int, int] | None = None
    """Interaction duration range."""
    duration_min: int | None = None
    """Interaction duration minimum."""
    duration_max: int | None = None
    """Interaction duration maximum."""
    duration_step: int | None = None
    """Interaction duration step."""
    high_qubit_amplitude: float | None = None
    """High frequency qubit amplitude."""
    dt: int | None = 0
    """Time delay between flux pulses and readout."""
    native: Literal["CZ", "iSWAP"] = "iSWAP"
    """Two qubit interaction to be calibrated."""

    @property
    def coupler_flux_amplitude_range(
        self,
    ) -> tuple[float, float, float] | tuple[None, None, None]:
        if self.coupler_ampl_range is None:
            return (
                self.coupler_ampl_min,
                self.coupler_ampl_max,
                self.coupler_ampl_step,
            )
        return self.coupler_ampl_range

    @property
    def interaction_duration_range(
        self,
    ) -> tuple[int, int, int] | tuple[None, None, None]:
        if self.duration_range is None:
            return (
                self.duration_min,
                self.duration_max,
                self.duration_step,
            )
        return self.duration_range

    def __post_init__(self) -> None:

        if any([x is None for x in self.interaction_duration_range]):
            raise InvalidInputParameters(
                "high frequency qubit amplitude range not set properly"
            )

        if any([x is None for x in self.coupler_flux_amplitude_range]):
            raise InvalidInputParameters("coupler amplitude range not set properly")

    def coupler_flux_amplitude_values(self) -> npt.NDArray:
        return np.arange(*self.coupler_flux_amplitude_range)

    def interaction_duration_values(self) -> npt.NDArray:
        return np.arange(*self.interaction_duration_range)


@dataclass
class CouplerChevronTimeResults(Results):
    """CouplerChevronTime outputs when fitting will be done."""

    interaction_times: list[float]
    """Rabi time values."""
    high_freq_ampl: float
    """Amplitude for high frequency qubit flux line."""
    native: Literal["CZ", "iSWAP"]
    """Two qubit interaction to be calibrated."""
    min_interaction_point: dict[QubitPairId, tuple[float, float, int]] = field(
        default_factory=tuple
    )
    coupling_strengths: dict[QubitPairId, dict[float, float]] = field(
        default_factory=dict
    )
    """Coupling constant as a function of the coupler flux amplitude."""
    gate_durations: dict[QubitPairId, dict[float, int]] = field(default_factory=dict)
    """Gate durations for different coupler flux amplitudes."""
    detunings: dict[QubitPairId, dict[float, int]] = field(default_factory=dict)
    """Detunings for different coupler flux amplitudes."""


CouplerChevronTimeType = np.dtype(
    [
        ("length", int),
        ("coupler_amplitude", np.float64),
        ("prob_high", np.float64),
        ("prob_low", np.float64),
    ]
)
"""Custom dtype for CouplerChevron."""


@dataclass
class CouplerChevronTimeData(Data):
    """CouplerChevronTime acquisition outputs."""

    native: Literal["CZ", "iSWAP"]
    """Two qubit interaction to be calibrated.
    iSWAP and CZ are the possible options.
    """
    coupler_ampl: list[float]
    """Offset values."""
    interaction_times: list[float]
    """Rabi time values."""
    high_freq_ampl: float
    """Amplitude for high frequency qubit flux line."""
    data: dict[QubitPairId, npt.NDArray[CouplerChevronTimeType]] = field(
        default_factory=dict
    )

    def register_qubit(
        self, low_qubit, high_qubit, length, coupler_amplitude, prob_low, prob_high
    ):
        """Store output for single qubit."""
        size = len(length) * len(coupler_amplitude)
        coupler_amplitude, durations = np.meshgrid(coupler_amplitude, length)
        ar = np.empty(size, dtype=CouplerChevronTimeType)
        ar["length"] = durations.ravel()
        ar["coupler_amplitude"] = coupler_amplitude.ravel()
        ar["prob_low"] = prob_low.ravel()
        # Since an X gate was added on the high frequency qubit at the end of the
        # pulse sequence, its CouplerChevronTime pattern is between state 0 and 2, so the state
        # one is mapped into 0. For this reason and compatibility with the other
        # qubit, we have to evaluate the ground state probability.
        if self.native == "CZ":
            ar["prob_high"] = 1 - prob_high.ravel()
        else:
            ar["prob_high"] = prob_high.ravel()
        self.data[low_qubit, high_qubit] = np.rec.array(ar)


def _aquisition(
    params: CouplerChevronTimeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerChevronTimeData:
    r"""Perform an two qubit interaction experiment between pairs of qubits by changing the in-between
    coupler frequency via sweeping over its DC flux line.
    """

    # create a DataUnits object to store the results
    data = CouplerChevronTimeData(
        native=params.native,
        coupler_ampl=params.coupler_flux_amplitude_values().tolist(),
        interaction_times=params.interaction_duration_values().tolist(),
        high_freq_ampl=params.high_qubit_amplitude,
    )

    couplers_map = platform.find_coupler_for_pair_list(targets)
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        low_freq_qubit, high_freq_qubit = order_pair(pair, platform)

        sequence = PulseSequence()

        low_natives = platform.natives.single_qubit[low_freq_qubit]
        low_drive_channel, low_drive_pulse = low_natives.RX()[0]
        ro_low_channel, ro_low_pulse = low_natives.MZ()[0]

        high_natives = platform.natives.single_qubit[high_freq_qubit]
        high_drive_channel, high_drive_pulse = high_natives.RX()[0]
        ro_high_channel, ro_high_pulse = high_natives.MZ()[0]

        drive_duration = high_drive_pulse.duration
        if params.native == "CZ":
            sequence += [(low_drive_channel, low_drive_pulse)]
            drive_duration = max(low_drive_pulse.duration, high_drive_pulse.duration)
        sequence += [(high_drive_channel, high_drive_pulse)]

        # delay for coupler flux line
        coupler_flux_channel = platform.couplers[couplers_map[pair]].flux
        sequence += [(coupler_flux_channel, Delay(duration=drive_duration))]
        coupler_flux_pulse = Pulse(
            duration=params.interaction_duration_range[1],
            amplitude=float(params.coupler_flux_amplitude_range[1]),
            relative_phase=0.0,
            envelope=Rectangular(),
        )
        # adding coupler flux pulse
        sequence += [(coupler_flux_channel, coupler_flux_pulse)]

        # delay for high freq qubit flux line
        high_flux_channel = platform.qubits[high_freq_qubit].flux
        sequence += [(high_flux_channel, Delay(duration=drive_duration))]
        high_flux_pulse = Pulse(
            duration=params.interaction_duration_range[1],
            amplitude=float(params.high_qubit_amplitude),
            relative_phase=0.0,
            envelope=Rectangular(),
        )
        # adding high freq qubit flux pulse
        sequence += [(high_flux_channel, high_flux_pulse)]

        delays = [Delay(duration=params.interaction_duration_range[1])] * 3
        dt_delay = Delay(duration=params.dt)
        sequence += [
            (ro_low_channel, Delay(duration=drive_duration)),
            (ro_high_channel, Delay(duration=drive_duration)),
            (high_drive_channel, delays[0]),
            (ro_low_channel, delays[1]),
            (ro_high_channel, delays[2]),
            (high_drive_channel, dt_delay),
            (ro_low_channel, dt_delay),
            (ro_high_channel, dt_delay),
        ]

        if params.native == "CZ":
            sequence += [
                (high_drive_channel, high_drive_pulse),
                (ro_low_channel, Delay(duration=high_drive_pulse.duration)),
                (ro_high_channel, Delay(duration=high_drive_pulse.duration)),
            ]

        # add readout
        sequence += [(ro_low_channel, ro_low_pulse), (ro_high_channel, ro_high_pulse)]

        # sweeper for coupler
        c_sweeper_amplitude = Sweeper(
            parameter=Parameter.amplitude,
            range=params.coupler_flux_amplitude_range,
            pulses=[coupler_flux_pulse],
        )
        coupler_sweepers = ParallelSweepers([c_sweeper_amplitude])

        # sweeper for flux pulse duration
        duration_parsweeps = ParallelSweepers([])
        for p in [coupler_flux_pulse, high_flux_pulse] + delays:
            duration_parsweeps.append(
                Sweeper(
                    parameter=Parameter.duration,
                    range=params.duration_range,
                    pulses=[p],
                )
            )

        ro_high = list(sequence.channel(platform.qubits[high_freq_qubit].acquisition))[
            -1
        ]
        ro_low = list(sequence.channel(platform.qubits[low_freq_qubit].acquisition))[-1]

        results = platform.execute(
            [sequence],
            [duration_parsweeps, coupler_sweepers],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        data.register_qubit(
            low_qubit=low_freq_qubit,
            high_qubit=high_freq_qubit,
            coupler_amplitude=params.coupler_flux_amplitude_values(),
            length=params.interaction_duration_values(),
            prob_low=results[ro_low.id],
            prob_high=results[ro_high.id],
        )
    return data


def _fit(data: CouplerChevronTimeData) -> CouplerChevronTimeResults:
    coupling_strengths: dict[QubitPairId, dict[float, float]] = {}
    gate_durations: dict[QubitPairId, dict[float, int]] = {}
    detunings: dict[QubitPairId, dict[float, int]] = {}
    min_interaction: dict[QubitPairId, tuple[float, float, int]] = {}
    for pair in data.data:
        pair_data = data.data[pair]

        try:
            for c_a in data.coupler_ampl:
                selected_amp = pair_data.coupler_amplitude == c_a
                times = pair_data[pair].length[selected_amp]
                ampl_data_high = pair_data.prob_high[selected_amp]
                ampl_data_low = pair_data.prob_low[selected_amp]

                high_popt, _ = curve_fit(
                    chevron_function,
                    times,
                    ampl_data_high,
                    p0=[0, 0],
                    bounds=(
                        [-np.inf, -np.inf],
                        [np.inf, np.inf],
                    ),
                )

                low_popt, _ = curve_fit(
                    1 - chevron_function,
                    times,
                    ampl_data_low,
                    p0=[0, 0],
                    bounds=(
                        [-np.inf, -np.inf],
                        [np.inf, np.inf],
                    ),
                )

                detuning = (high_popt[0] + low_popt[0]) / 2
                g = (high_popt[1] + low_popt[1]) / 2
                period = 2 * np.pi / (detuning**2 + 4 * g**2)
                duration = period if data.native == "CZ" else period / 2

                coupling_strengths.setdefault(pair, {})[c_a] = float(g)
                gate_durations.setdefault(pair, {})[c_a] = int(duration)
                detunings.setdefault(pair, {})[c_a] = int(detuning)

            g_vs_ampl = list(coupling_strengths[pair].values())
            min_g = np.min(np.abs(g_vs_ampl))
            min_zz_indices = np.where(g_vs_ampl == min_g)[0]
            ampl_list = list(coupling_strengths[pair].keys())
            min_g_ampl = np.min(np.abs(ampl_list)[min_zz_indices])
            min_g_ampl_index = np.where(np.abs(ampl_list) == min_g_ampl)[0]

            min_g = float(g_vs_ampl[min_g_ampl_index[0]])
            sel_ampl = ampl_list[min_g_ampl_index[0]]
            sel_gate_t = gate_durations[pair][sel_ampl]
            min_interaction[pair] = (sel_ampl, min_g, sel_gate_t)

        except Exception as e:
            log.warning(f"CouplerChevronTime fit failed for pair {pair} due to {e}")

    return CouplerChevronTimeResults(
        interaction_times=data.interaction_times,
        high_freq_ampl=data.high_freq_ampl,
        native=data.native,
        coupling_strengths=coupling_strengths,
        gate_durations=gate_durations,
        detunings=detunings,
        min_interaction_point=min_interaction,
    )


def _plot(
    data: CouplerChevronTimeData, fit: CouplerChevronTimeResults, target: QubitPairId
):
    """Plot the experiment result for a single pair."""
    # reverse qubit order if not found in data
    if target not in data.data:
        target = (target[1], target[0])

    pair_data = data.data[target]
    figures = []

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {target[0]} - Low Frequency",
            f"Qubit {target[1]} - High Frequency",
        ),
    )
    fitting_report = ""

    fig.add_trace(
        go.Heatmap(
            x=pair_data.length,
            y=pair_data.coupler_amplitude,
            z=pair_data.prob_low,
            coloraxis=COLORAXIS[0],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=pair_data.length,
            y=pair_data.coupler_amplitude,
            z=pair_data.prob_high,
            coloraxis=COLORAXIS[1],
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        xaxis_title="Duration [ns]",
        xaxis2_title="Duration [ns]",
        yaxis_title="Amplitude [a.u.]",
        legend=dict(orientation="h"),
    )
    fig.update_layout(
        coloraxis={"colorscale": "Oryel", "colorbar": {"x": 1.15}},
        coloraxis2={"colorscale": "Darkmint", "colorbar": {"x": -0.15}},
    )

    figures.append(fig)

    if (
        fit is not None
        and target in fit.coupling_strengths
        and target in fit.gate_durations
    ):
        interaction_fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Coupling Strength", "Gate duration")
        )

        interaction_fig.add_trace(
            go.Scatter(
                x=list(fit.coupling_strengths[target].keys()),
                y=np.array([fit.coupling_strengths[target].values()]) * HZ_TO_MHZ,
                name="g coupling",
                line=go.scatter.Line(dash="dashdot"),
            ),
            row=1,
            col=1,
        )

        interaction_fig.add_trace(
            go.Scatter(
                x=list(fit.gate_durations[target].keys()),
                y=np.array([fit.gate_durations[target].values()]) * HZ_TO_MHZ,
                name="gate duration",
                line=go.scatter.Line(dash="dashdot"),
            ),
            row=1,
            col=2,
        )

        if target in fit.min_interaction_point:
            interaction_fig.add_trace(
                go.Scatter(
                    x=fit.min_interaction_point[target][0],
                    y=fit.min_interaction_point[target][1] * HZ_TO_MHZ,
                    name="min g coupling",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            interaction_fig.add_trace(
                go.Scatter(
                    x=fit.min_interaction_point[target][0],
                    y=fit.min_interaction_point[target][2],
                    name="min g gate duration",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=2,
            )

            fitting_report = table_html(
                table_dict(
                    [target] * 3,
                    [
                        f"{fit.native} minimum g flux ampl [a.u.]",
                        f"{fit.native} minimum g [MHz]",
                        f"{fit.native} minimum g gate duration [ns]",
                    ],
                    [
                        fit.min_interaction_point[target][1] * HZ_TO_MHZ,
                        fit.min_interaction_point[target][2],
                    ],
                )
            )

    return figures, fitting_report


def _update(
    results: CouplerChevronTimeResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    # target = target[::-1] if target not in results.duration else target

    # getattr(update, f"{results.native}_duration")(
    #     results.duration[target], platform, target
    # )
    # getattr(update, f"{results.native}_amplitude")(
    #     results.amplitude[target], platform, target
    # )
    pass


coupler_chevron_time = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""CouplerChevronTime routine."""
