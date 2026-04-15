from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
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
from qibocal.protocols.two_qubit_interaction.chevron.utils import (
    COLORAXIS,
    chevron_fit,
)
from qibocal.protocols.two_qubit_interaction.utils import fit_flux_amplitude, order_pair
from qibocal.protocols.utils import table_dict, table_html

__all__ = ["coupler_chevron_amplitude"]


class InvalidInputParameters(Exception):
    pass


@dataclass
class CouplerChevronAmplParameters(Parameters):
    """CouplerChevronAmpl runcard inputs."""

    duration: int
    """Interaction duration."""
    coupler_ampl_range: tuple[float, float, float] | None = None
    """Coupler amplitude range."""
    coupler_ampl_min: float | None = None
    """Coupler amplitude minimum."""
    coupler_ampl_max: float | None = None
    """Coupler amplitude maximum."""
    coupler_ampl_step: float | None = None
    """Coupler amplitude step."""
    high_qubit_ampl_range: tuple[float, float, float] | None = None
    """High frequency qubit amplitude range."""
    high_qubit_ampl_min: float | None = None
    """High frequency qubit amplitude minimum."""
    high_qubit_ampl_max: float | None = None
    """High frequency qubit amplitude maximum."""
    high_qubit_ampl_step: float | None = None
    """High frequency qubit amplitude step."""
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
    def high_qubit_flux_amplitude_range(
        self,
    ) -> tuple[float, float, float] | tuple[None, None, None]:
        if self.high_qubit_ampl_range is None:
            return (
                self.high_qubit_ampl_min,
                self.high_qubit_ampl_max,
                self.high_qubit_ampl_step,
            )
        return self.high_qubit_ampl_range

    def __post_init__(self) -> None:

        if any([x is None for x in self.high_qubit_flux_amplitude_range]):
            raise InvalidInputParameters(
                "high frequency qubit amplitude range not set properly"
            )

        if any([x is None for x in self.coupler_flux_amplitude_range]):
            raise InvalidInputParameters("coupler amplitude range not set properly")

    def coupler_flux_amplitude_values(self) -> npt.NDArray:
        return np.arange(*self.coupler_flux_amplitude_range)

    def high_qubit_flux_amplitude_values(self) -> npt.NDArray:
        return np.arange(*self.high_qubit_flux_amplitude_range)


@dataclass
class CouplerChevronAmplResults(Results):
    """CouplerChevronAmpl outputs when fitting will be done."""

    duration: int
    """Interaction duration."""
    coupler_ampl: dict[QubitPairId, float]
    """CZ angle."""
    qubit_ampl: dict[QubitPairId, float]
    """Virtual Z phase correction."""
    native: Literal["CZ", "iSWAP"] = "iSWAP"
    """Two qubit interaction to be calibrated."""


CouplerChevronAmplType = np.dtype(
    [
        ("coupler_amplitude", np.float64),
        ("qubit_amplitude", np.float64),
        ("prob_high", np.float64),
        ("prob_low", np.float64),
    ]
)
"""Custom dtype for CouplerChevron."""


@dataclass
class CouplerChevronAmplData(Data):
    """CouplerChevronAmpl acquisition outputs."""

    native: str
    """Two qubit interaction to be calibrated.
    iSWAP and CZ are the possible options.
    """
    int_duration: float
    """Rabi time values."""
    coupler_ampl: list[float]
    """Coupler amplitude values."""
    qubit_ampl: list[float]
    """High frequency qubit amplitude values."""
    data: dict[QubitPairId, npt.NDArray[CouplerChevronAmplType]] = field(
        default_factory=dict
    )
    """Raw data."""

    def register_qubit(
        self,
        low_qubit,
        high_qubit,
        coupler_amplitude,
        qubit_amplitude,
        prob_low,
        prob_high,
    ):
        """Store output for single qubit."""
        size = len(qubit_amplitude) * len(coupler_amplitude)
        qubit_amplitude, coupler_amplitude = np.meshgrid(
            qubit_amplitude, coupler_amplitude
        )
        ar = np.empty(size, dtype=CouplerChevronAmplType)
        ar["coupler_amplitude"] = coupler_amplitude.ravel()
        ar["qubit_amplitude"] = qubit_amplitude.ravel()
        ar["prob_low"] = prob_low.ravel()
        # Since an X gate was added on the high frequency qubit at the end of the
        # pulse sequence, its CouplerChevronAmpl pattern is between state 0 and 2, so the state
        # one is mapped into 0. For this reason and compatibility with the other
        # qubit, we have to evaluate the ground state probability.
        if self.native == "CZ":
            ar["prob_high"] = 1 - prob_high.ravel()
        else:
            ar["prob_high"] = prob_high.ravel()
        self.data[low_qubit, high_qubit] = np.rec.array(ar)

    def low_frequency(self, pair):
        return self[pair].prob_low

    def high_frequency(self, pair):
        return self[pair].prob_high


def _aquisition(
    params: CouplerChevronAmplParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerChevronAmplData:
    r"""Perform an two qubit interaction experiment between pairs of qubits by changing the in-between
    coupler frequency via sweeping over its DC flux line.
    """

    # create a DataUnits object to store the results
    data = CouplerChevronAmplData(
        native=params.native,
        int_duration=params.duration,
        coupler_ampl=params.coupler_flux_amplitude_values().tolist(),
        qubit_ampl=params.high_qubit_flux_amplitude_values().tolist(),
    )

    couplers_map = platform.find_coupler_for_pair_list(targets)
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        low_freq_qubit, high_freq_qubit = order_pair(pair, platform)

        sequence = PulseSequence()

        # low frequency qubit channels
        low_natives = platform.natives.single_qubit[low_freq_qubit]
        low_drive_channel, low_drive_pulse = low_natives.RX()[0]
        ro_low_channel, ro_low_pulse = low_natives.MZ()[0]

        # high frequency qubit channels
        high_natives = platform.natives.single_qubit[high_freq_qubit]
        high_drive_channel, high_drive_pulse = high_natives.RX()[0]
        ro_high_channel, ro_high_pulse = high_natives.MZ()[0]

        drive_duration = high_drive_pulse.duration
        if params.native == "CZ":
            sequence.append((low_drive_channel, low_drive_pulse))
            drive_duration = max(low_drive_pulse.duration, drive_duration)
        sequence.append((high_drive_channel, high_drive_pulse))

        # delay for coupler flux line
        coupler_flux_channel = platform.couplers[couplers_map[pair]].flux
        coupler_flux_pulse = Pulse(
            duration=params.duration,
            amplitude=float(params.coupler_flux_amplitude_range[1]),
            relative_phase=0.0,
            envelope=Rectangular(),
        )

        # delay for high freq qubit flux line
        high_flux_channel = platform.qubits[high_freq_qubit].flux
        high_flux_pulse = Pulse(
            duration=params.duration,
            amplitude=float(params.high_qubit_flux_amplitude_range[1]),
            relative_phase=0.0,
            envelope=Rectangular(),
        )

        # adding coupler and high freq qubit flux pulses
        sequence |= PulseSequence(
            [
                (coupler_flux_channel, coupler_flux_pulse),
                (high_flux_channel, high_flux_pulse),
            ]
        )

        # add readout
        sequence |= PulseSequence(
            [(ro_low_channel, ro_low_pulse), (ro_high_channel, ro_high_pulse)]
        )

        # sweeper for coupler
        c_sweeper_amplitude = Sweeper(
            parameter=Parameter.amplitude,
            range=params.coupler_ampl_range,
            pulses=[coupler_flux_pulse],
        )
        coupler_sweepers = ParallelSweepers([c_sweeper_amplitude])

        # sweeper for high frequency qubit
        q_sweeper_amplitude = Sweeper(
            parameter=Parameter.amplitude,
            range=params.high_qubit_ampl_range,
            pulses=[high_flux_pulse],
        )
        high_qubit_sweepers = ParallelSweepers([q_sweeper_amplitude])

        ro_high = list(sequence.channel(platform.qubits[high_freq_qubit].acquisition))[
            -1
        ]
        ro_low = list(sequence.channel(platform.qubits[low_freq_qubit].acquisition))[-1]

        results = platform.execute(
            [sequence],
            [coupler_sweepers, high_qubit_sweepers],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        data.register_qubit(
            low_qubit=low_freq_qubit,
            high_qubit=high_freq_qubit,
            coupler_amplitude=params.coupler_flux_amplitude_values(),
            qubit_amplitude=params.high_qubit_flux_amplitude_values(),
            prob_low=results[ro_low.id],
            prob_high=results[ro_high.id],
        )
    return data


def _fit(data: CouplerChevronAmplData) -> CouplerChevronAmplResults:
    coupler_amplitudes = {}
    high_freq_amplitudes = {}
    for pair in data.data:
        c_amps = data.coupler_ampl
        q_amps = data.qubit_ampl

        signal = data.low_frequency(pair)
        signal_matrix = signal.reshape(len(q_amps), len(c_amps)).T

        # guess amplitude computing FFT
        amplitude, index, delta = fit_flux_amplitude(signal_matrix, c_amps, q_amps)
        # estimate duration by rabi curve at amplitude previously estimated
        y = signal_matrix[index, :].ravel()
        try:
            popt, _ = curve_fit(
                chevron_fit,
                q_amps,
                y,
                p0=[delta * 2 * np.pi, np.pi, np.mean(y), np.mean(y)],
                bounds=(
                    [0, -2 * np.pi, np.min(y), np.min(y)],
                    [np.inf, 2 * np.pi, np.max(y), np.max(y)],
                ),
            )
            coupler_amplitudes[pair] = amplitude
            high_freq_amplitudes[pair] = q_amps
        except Exception as e:
            log.warning(f"CouplerChevronAmpl fit failed for pair {pair} due to {e}")

    return CouplerChevronAmplResults(
        duration=data.int_duration,
        coupler_ampl=coupler_amplitudes,
        qubit_ampl=high_freq_amplitudes,
        native=data.native,
    )


def _plot(
    data: CouplerChevronAmplData, fit: CouplerChevronAmplResults, target: QubitPairId
):
    """Plot the experiment result for a single pair."""
    # reverse qubit order if not found in data
    if target not in data.data:
        target = (target[1], target[0])

    pair_data = data[target]
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
            x=pair_data.coupler_amplitude,
            y=pair_data.qubit_amplitude,
            z=data.low_frequency(target),
            coloraxis=COLORAXIS[0],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=pair_data.coupler_amplitude,
            y=pair_data.qubit_amplitude,
            z=data.high_frequency(target),
            coloraxis=COLORAXIS[1],
        ),
        row=1,
        col=2,
    )

    for measured_qubit in target:
        if fit is not None:
            fig.add_trace(
                go.Scatter(
                    x=[
                        fit.qubit_ampl[target],
                    ],
                    y=[
                        fit.coupler_ampl[target],
                    ],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="black",
                        symbol="cross",
                    ),
                    name=f"{data.native} estimate",  #  Change name from the params
                    showlegend=True if measured_qubit == target[0] else False,
                    legendgroup="Voltage",
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        xaxis_title="Qubit flux amplitude [a.u.]",
        xaxis2_title="Coupler flux amplitude [a.u.]",
        yaxis_title="Amplitude [a.u.]",
        legend=dict(orientation="h"),
    )
    fig.update_layout(
        coloraxis={"colorscale": "Oryel", "colorbar": {"x": 1.15}},
        coloraxis2={"colorscale": "Darkmint", "colorbar": {"x": -0.15}},
    )

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                target[1],
                [f"{fit.native} amplitude", f"{fit.native} high freq. qubit amplitude"],
                [
                    fit.coupler_ampl[target],
                    fit.qubit_ampl[target],
                ],
            )
        )

    return [fig], fitting_report


def _update(
    results: CouplerChevronAmplResults,
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


coupler_chevron_amplitude = Routine(
    _aquisition, _fit, _plot, _update, two_qubit_gates=True
)
"""CouplerChevronAmpl routine."""
