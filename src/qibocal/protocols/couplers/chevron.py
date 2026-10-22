"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    GaussianSquare,
    Parameter,
    Pulse,
    PulseId,
    PulseLike,
    PulseSequence,
    Sweeper,
)
from scipy.optimize import root

from qibocal import update
from qibocal.auto.operation import (
    DATAFILE,
    Data,
    Parameters,
    Protocol,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.two_qubit_interaction.chevron.utils import (
    chevron_fit,
)
from qibocal.protocols.two_qubit_interaction.utils import order_pair
from qibocal.protocols.utils import (
    Range,
    RangeLike,
    table_dict,
    table_html,
    to_range,
)

__all__ = ["coupler_chevron"]


@dataclass
class CouplerChevronParameters(Parameters):
    """CzFluxTime runcard inputs."""

    coupler_amplitude: RangeLike
    """Coupler amplitude range."""
    duration: RangeLike
    """Duration range."""
    dt: int | None = 0
    """Time delay between flux pulses and readout."""
    native: Literal["CZ", "iSWAP"] = "CZ"
    """Two qubit interaction to be calibrated."""

    @property
    def _coupler_amplitude(self) -> Range:
        return to_range(self.coupler_amplitude)

    @property
    def _duration(self) -> Range:
        return to_range(self.duration)


@dataclass
class CouplerChevronResults(Results):
    """CzFluxTime outputs when fitting will be done."""

    native: str = "CZ"
    """Two qubit interaction to be calibrated."""
    duration: dict[QubitPairId, list] = field(default_factory=dict)
    """Gate duration."""
    half_duration: dict[QubitPairId, list] = field(default_factory=dict)
    """Half gate duration."""
    amplitude: dict[QubitPairId, list] = field(default_factory=dict)
    """Gate flux amplitude."""
    fitted_parameters: dict[QubitPairId, list] = field(default_factory=dict)
    """Fitted parameters for chevron pattern."""


@dataclass
class CouplerChevronData(Data):
    """Chevron acquisition outputs."""

    native: str
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """
    duration: np.ndarray
    """Duration range."""
    amplitude: np.ndarray
    """Amplitude range."""
    _sorted_pairs: list = field(init=False, repr=False, metadata={"exclude": True})
    """Pairs to be calibrated sorted by frequency."""
    detuning: dict[QubitPairId, float] = field(default_factory=dict)
    """Expected detuning between qubit in pair."""
    anharmonicity: dict[QubitPairId, float] = field(default_factory=dict)
    """Anhamonicity of high frequency qubit in pair."""
    flux_coefficient: dict[QubitPairId, float] = field(default_factory=dict)
    """Flux coefficient to map frequency to amplitude."""
    data: dict[QubitPairId, np.ndarray] = field(default_factory=dict)
    """Raw data acquired."""

    label: str | None = None
    """Label for the data."""

    @property
    def grid(self) -> np.ndarray:
        x, y = np.meshgrid(self.duration, self.amplitude)
        return np.stack([x.ravel(), y.ravel()])

    @property
    def sorted_pairs(self) -> list:
        return [
            pair if isinstance(pair, tuple) else tuple(pair)
            for pair in self._sorted_pairs
        ]

    @sorted_pairs.setter
    def sorted_pairs(self, value):
        self._sorted_pairs = value

    @classmethod
    def load(cls, path: Path):
        """Generic load method."""
        data_dict = cls.load_data(path, DATAFILE)
        params = cls.load_params(path, DATAFILE)
        pairs = params.pop("_sorted_pairs")
        if data_dict is not None:
            if params is not None:
                instance = cls(data=data_dict, **params)
            else:
                instance = cls(data=data_dict)
        elif params is not None:
            instance = cls(**params)
        instance.sorted_pairs = pairs
        return instance


def _aquisition(
    params: CouplerChevronParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerChevronData:
    r"""Perform an CZ experiment between pairs of qubits by changing its
    frequency.

    Args:
        platform: CalibrationPlatform to use.
        params: Experiment parameters.
        targets (list): List of pairs to use sequentially.

    Returns:
        CouplerChevronData: Acquisition data.
    """

    sorted_pairs = [order_pair(pair, platform) for pair in targets]

    coupler_pulse = Pulse(
        duration=params._duration[0],
        amplitude=params._coupler_amplitude[0],
        envelope=GaussianSquare(
            risefall=0,
            sigma=0.2,
        ),
    )

    acq_delay = Delay(duration=params._duration[0])

    coupler_offsets: dict[QubitPairId, float] = {}
    high_freq_ro_ids: dict[QubitPairId, PulseId] = {}
    low_freq_ro_ids: dict[QubitPairId, PulseId] = {}
    time_swept_pulses: list[PulseLike] = [coupler_pulse, acq_delay]
    full_sequence = PulseSequence()
    for pair in sorted_pairs:
        flux_dep_func = platform.calibration.single_qubits[pair[1]].qubit.detuning
        low_frequency = platform.config(platform.qubits[pair[0]].drive).frequency

        def func(x, flux_dep_func=flux_dep_func, low_frequency=low_frequency):
            return flux_dep_func(x) - low_frequency

        amp_roots = root(func, 0)
        if not amp_roots.success or not np.isfinite(amp_roots.x[0]):
            raise ValueError(
                f"Could not find the resonant amplitude for qubit pair {pair}: "
                f"{amp_roots.message}"
            )
        res_amp = float(amp_roots.x[0])

        low_freq_natives = platform.parameters.native_gates.single_qubit[pair[0]]
        high_freq_natives = platform.parameters.native_gates.single_qubit[pair[1]]

        sequence = low_freq_natives.RX()
        if params.native == "CZ":
            sequence += high_freq_natives.RX()

        # adding high frequency flux pulse
        high_freq_flux_ch = platform.qubits[pair[1]].flux
        assert high_freq_flux_ch is not None, (
            "High frequency flux channel should not be None"
        )

        high_freq_flux = Pulse(
            duration=sequence.duration,
            amplitude=res_amp,
            envelope=GaussianSquare(
                risefall=0,
                sigma=0.2,
            ),
        )
        time_swept_pulses.append(high_freq_flux)

        coupler_id = platform.calibration.two_qubits[pair].coupler
        if coupler_id is None:
            raise ValueError("Coupler ID should not be None")

        coupler_channel = platform.couplers[coupler_id].flux
        coupler_offsets[pair] = platform.config(coupler_channel).offset
        assert coupler_channel is not None, "Coupler channel should not be None"

        low_freq_acq_ch, low_freq_acq_pulse = low_freq_natives.MZ()[0]
        high_freq_acq_ch, high_freq_acq_pulse = high_freq_natives.MZ()[0]

        low_freq_ro_ids[pair] = low_freq_acq_pulse.id
        high_freq_ro_ids[pair] = high_freq_acq_pulse.id

        sequence |= PulseSequence(
            [
                (high_freq_flux_ch, high_freq_flux),
                (coupler_channel, coupler_pulse),
                (low_freq_acq_ch, acq_delay),
                (high_freq_acq_ch, acq_delay),
            ]
        )

        sequence |= PulseSequence(
            [
                (low_freq_acq_ch, low_freq_acq_pulse),
                (high_freq_acq_ch, high_freq_acq_pulse),
            ]
        )

        full_sequence += sequence

    sweeper_duration = Sweeper(
        parameter=Parameter.duration,
        range=params._duration,
        pulses=time_swept_pulses,
    )

    sweeper_amplitude = Sweeper(
        parameter=Parameter.amplitude,
        range=params._coupler_amplitude,
        pulses=[coupler_pulse],
    )

    results = platform.execute(
        [full_sequence],
        [[sweeper_duration], [sweeper_amplitude]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = CouplerChevronData(
        native=params.native,
        duration=np.array(sweeper_duration.values),
        amplitude=np.array(sweeper_amplitude.values),
    )
    data.sorted_pairs = (sorted_pairs,)

    for pair in targets:
        data.data[pair] = np.stack(
            [
                results[low_freq_ro_ids[pair]],
                1 - results[high_freq_ro_ids[pair]]
                if params.native == "CZ"
                else results[high_freq_ro_ids[pair]],
            ]
        )
    return data


def _fit(data: CouplerChevronData) -> CouplerChevronResults:

    return CouplerChevronResults(native=data.native)


def _plot(data: CouplerChevronData, fit: CouplerChevronResults, target: QubitPairId):
    """Plot the experiment result for a single pair."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Qubit {target[0]} - Low Frequency - Data",
            f"Qubit {target[1]} - High Frequency - Data",
            f"Qubit {target[0]} - Low Frequency - minmax scaled",
            f"Qubit {target[1]} - High Frequency - minmax scaled",
        ),
        shared_xaxes="all",
        shared_yaxes="all",
    )
    fitting_report = ""
    for i, _data in enumerate([data[target][0], data[target][1]]):
        fig.add_trace(
            go.Heatmap(
                x=data.duration,
                y=data.amplitude,
                z=_data.T,
                coloraxis="coloraxis",
            ),
            row=1,
            col=1 + i,
        )

    if fit is not None:
        for i in range(2):
            if len(fit.fitted_parameters[target][i]) > 0:
                fig.add_trace(
                    go.Heatmap(
                        x=data.duration,
                        y=data.amplitude,
                        z=chevron_fit(
                            data.grid, *fit.fitted_parameters[target][i]
                        ).reshape(
                            len(data.amplitude),
                            len(data.duration),
                        ),
                        coloraxis="coloraxis",
                    ),
                    row=2,
                    col=i + 1,
                )

                for j in range(2):
                    fig.add_hline(
                        y=fit.amplitude[target][i],
                        line_dash="dot",
                        row=j + 1,
                        col=i + 1,
                        name="Fit",
                    )

                    fig.add_vline(
                        x=fit.duration[target][i],
                        line_dash="dot",
                        row=j + 1,
                        col=i + 1,
                        name="Fit",
                    )

                    fig.add_vline(
                        x=fit.half_duration[target][i],
                        line_dash="dot",
                        row=j + 1,
                        col=i + 1,
                        name="Fit",
                    )

    fig.update_layout(
        xaxis_title="Duration [ns]",
        xaxis2_title="Duration [ns]",
        xaxis3_title="Duration [ns]",
        xaxis4_title="Duration [ns]",
        yaxis_title="Amplitude [a.u.]",
        yaxis2_title="Amplitude [a.u.]",
        yaxis3_title="Amplitude [a.u.]",
        yaxis4_title="Amplitude [a.u.]",
        legend={"orientation": "h"},
        coloraxis={
            "colorscale": "Inferno",
            # Fix colorbar range to [0, 1]: the plotted values are probabilities.
            "cmin": 0,
            "cmax": 1,
            "colorbar": {"title": {"text": "P<sub>e</sub>"}},
        },
        height=800,
    )
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(showticklabels=True, showline=True, row=i, col=j)
            fig.update_yaxes(showticklabels=True, showline=True, row=i, col=j)

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                target[1],
                [f"{fit.native} amplitude", f"{fit.native} duration"],
                [
                    np.mean(fit.amplitude[target]),
                    np.round(np.mean(fit.duration[target]))
                    if data.native == "CZ"
                    else np.round(np.mean(fit.half_duration[target])),
                ],
            )
        )

    return [fig], fitting_report


def _update(
    results: CouplerChevronResults, platform: CalibrationPlatform, target: QubitPairId
):
    # Fit results use the acquisition order, while native gates use the platform's pair
    # order.
    reversed_target = (target[1], target[0])
    result_target = target if target in results.duration else reversed_target
    platform_target = (
        target if target in platform.natives.two_qubit else reversed_target
    )

    getattr(update, f"{results.native}_duration")(
        np.mean(results.duration[result_target]), platform, platform_target
    )
    getattr(update, f"{results.native}_amplitude")(
        np.mean(results.amplitude[result_target]), platform, platform_target
    )


coupler_chevron = Protocol(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
