"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper
from scipy.optimize import curve_fit

from qibocal.auto.operation import (
    DATAFILE,
    Data,
    Parameters,
    Protocol,
    QubitId,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    Range,
    RangeLike,
    minmax_scaling,
    table_dict,
    table_html,
    to_range,
)

from .... import update
from ..utils import order_pair
from .utils import chevron_fit, chevron_sequence, cz_level2_qubit

__all__ = ["chevron"]


@dataclass
class ChevronParameters(Parameters):
    """CzFluxTime runcard inputs."""

    amplitude: RangeLike | None = None
    """Amplitude range."""
    duration: RangeLike | None = None
    """Duration range."""
    amplitude_min: float | None = None
    """Amplitude minimum. Deprecated, use :ref:`amplitude` instead."""
    amplitude_max: float | None = None
    """Amplitude maximum. Deprecated, use :ref:`amplitude` instead."""
    amplitude_step: float | None = None
    """Amplitude step. Deprecated, use :ref:`amplitude` instead."""
    duration_min: float | None = None
    """Duration minimum. Deprecated, use :ref:`duration` instead."""
    duration_max: float | None = None
    """Duration maximum. Deprecated, use :ref:`duration` instead."""
    duration_step: float | None = None
    """Duration step. Deprecated, use :ref:`duration` instead."""
    dt: int = 0
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Whether to park non interacting qubits or not."""
    native: Literal["CZ", "iSWAP"] = "CZ"
    """Two qubit interaction to be calibrated."""
    infer_hybridization: bool = False
    """Infer the CZ interaction branch from the calibration data.

    By default, the CZ interaction branch is the one selected by ``cz_branch``.
    If ``True``, the calibrated frequencies, anharmonicities and flux coefficients
    are used to determine whether the flux pulse reaches the ``|11>-|02>`` or the
    ``|11>-|20>`` resonance first, within the scanned amplitude range.
    The final RX pulse is applied to the qubit occupying level 2, whose
    population is inverted, and the fit is initialized accordingly.
    Ignored for iSWAP.
    """
    cz_branch: Literal["11-02", "11-20"] = "11-02"
    """CZ interaction branch, used if ``infer_hybridization`` is not set.

    States are labelled as ``|low, high>``, with qubits ordered by frequency, so
    that level 2 is occupied by the high frequency qubit for ``11-02``, and by the
    low frequency one for ``11-20``. Ignored for iSWAP.
    """

    @property
    def amplitude_range(self) -> Range:
        if self.amplitude is not None:
            return to_range(self.amplitude)

        if any(
            a is None
            for a in (self.amplitude_min, self.amplitude_max, self.amplitude_step)
        ):
            raise ValueError("Amplitude range is not properly defined.")

        return (self.amplitude_min, self.amplitude_max, self.amplitude_step)

    @property
    def amplitude_values(self) -> np.ndarray:
        return np.arange(*self.amplitude_range)

    @property
    def duration_range(self) -> Range:
        if self.duration is not None:
            return to_range(self.duration)

        if any(
            d is None
            for d in (self.duration_min, self.duration_max, self.duration_step)
        ):
            raise ValueError("Duration range is not properly defined.")

        return (self.duration_min, self.duration_max, self.duration_step)

    @property
    def duration_values(self) -> np.ndarray:
        return np.arange(*self.duration_range)


@dataclass
class ChevronResults(Results):
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
class ChevronData(Data):
    """Chevron acquisition outputs."""

    native: str
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """
    amplitude: list
    """Amplitude values."""
    duration: list
    """Duration values."""
    _sorted_pairs: list = field(init=False, repr=False, metadata={"exclude": True})
    """Pairs to be calibrated sorted by frequency."""
    detuning: dict[QubitPairId, float] = field(default_factory=dict)
    """Expected detuning between qubit in pair."""
    anharmonicity: dict[QubitPairId, float] = field(default_factory=dict)
    """Anharmonicity term of the detuning at the interaction, in GHz.

    The anharmonicity of the high frequency qubit for ``|11>-|02>``, the opposite
    of the low frequency qubit one for ``|11>-|20>``, and 0 for iSWAP.
    """
    level2_qubit: dict[QubitPairId, QubitId] = field(default_factory=dict)
    """Qubit occupying level 2 in the CZ interaction, not defined for iSWAP."""
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
    params: ChevronParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> ChevronData:
    r"""Perform an CZ experiment between pairs of qubits by changing its
    frequency.

    Args:
        platform: CalibrationPlatform to use.
        params: Experiment parameters.
        targets (list): List of pairs to use sequentially.

    Returns:
        ChevronData: Acquisition data.
    """

    data = ChevronData(
        native=params.native,
        amplitude=params.amplitude_values.tolist(),
        duration=params.duration_values.tolist(),
    )

    data.sorted_pairs = [order_pair(pair, platform) for pair in targets]
    data.flux_coefficient = {
        pair: platform.calibration.single_qubits[pair[1]].qubit.flux_coefficients[0]
        for pair in data.sorted_pairs
    }
    data.detuning = {
        pair: (
            platform.calibration.single_qubits[pair[1]].qubit.frequency_01
            - platform.calibration.single_qubits[pair[0]].qubit.frequency_01
        )
        * HZ_TO_GHZ
        for pair in data.sorted_pairs
    }
    _set_interaction_branch(params, platform, data)

    for pair in data.sorted_pairs:
        sequence, flux_pulse, parking_pulses, delays = chevron_sequence(
            platform=platform,
            ordered_pair=pair,
            duration_max=params.duration_range[1],
            parking=params.parking,
            dt=params.dt,
            native=params.native,
            level2_qubit=data.level2_qubit.get(pair),
        )
        sweeper_amplitude = Sweeper(
            parameter=Parameter.amplitude,
            range=params.amplitude_range,
            pulses=[flux_pulse],
        )
        sweeper_duration = Sweeper(
            parameter=Parameter.duration,
            range=params.duration_range,
            pulses=[flux_pulse] + delays + parking_pulses,
        )

        ro_high = list(sequence.channel(platform.qubits[pair[1]].acquisition))[-1]
        ro_low = list(sequence.channel(platform.qubits[pair[0]].acquisition))[-1]

        results = platform.execute(
            [sequence],
            [[sweeper_duration], [sweeper_amplitude]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        data.data[pair[0], pair[1]] = invert_level2(
            data, pair, [results[ro_low.id], results[ro_high.id]]
        )
    return data


def _set_interaction_branch(
    params: ChevronParameters, platform: CalibrationPlatform, data: ChevronData
):
    """Store the qubit occupying level 2 and the anharmonicity term of each pair.

    The detuning between the interacting states at zero amplitude, used to
    initialize the fit, is ``f_high - f_low + anharmonicity``, with
    ``anharmonicity`` being ``0`` for ``|10>-|01>`` (iSWAP), ``alpha_high`` for
    ``|11>-|02>`` and ``-alpha_low`` for ``|11>-|20>``.
    """
    amplitude_range = (min(data.amplitude), max(data.amplitude))
    for pair in data.sorted_pairs:
        level2_qubit = cz_level2_qubit(
            platform,
            pair,
            native=params.native,
            infer_hybridization=params.infer_hybridization,
            cz_branch=params.cz_branch,
            amplitude_range=amplitude_range,
        )
        data.anharmonicity[pair] = 0.0
        if level2_qubit is not None:
            data.level2_qubit[pair] = level2_qubit
            sign = 1 if level2_qubit == pair[1] else -1
            data.anharmonicity[pair] = (
                sign
                * platform.calibration.single_qubits[level2_qubit].qubit.anharmonicity
                * HZ_TO_GHZ
            )


def invert_level2(
    data: ChevronData, pair: QubitPairId, signals: list[np.ndarray]
) -> np.ndarray:
    """Stack the low and high qubit signals, inverting the one of level 2 qubit.

    The final RX pulse maps the initial ``|1>`` state of the qubit occupying level 2
    to ``|0>``, while it leaves ``|2>`` unchanged, so that its inverted population
    oscillates in phase with the other qubit one.
    """
    level2_qubit = data.level2_qubit.get(pair)
    return np.stack(
        [
            1 - signal if qubit == level2_qubit else signal
            for qubit, signal in zip(pair, signals)
        ]
    )


def _resonance_amplitude(amplitude: float, amplitudes: list[float]) -> float:
    """Signed resonance amplitude, given its absolute value.

    The fit model only depends on the squared amplitude, hence the negative
    resonance is selected if only that one lies within the scanned amplitudes.
    """

    def scanned(value):
        return min(amplitudes) <= value <= max(amplitudes)

    return -amplitude if scanned(-amplitude) and not scanned(amplitude) else amplitude


def _fit(data: ChevronData) -> ChevronResults:
    fitted_parameters = {}
    duration = {}
    amplitude = {}
    half_duration = {}
    for pair in data.sorted_pairs:
        fitted_parameters[pair] = []
        duration[pair], half_duration[pair], amplitude[pair] = [], [], []
        for _data in [data[pair][0], data[pair][1]]:
            try:
                popt, _ = curve_fit(
                    chevron_fit,
                    data.grid,
                    minmax_scaling(_data.T.flatten(), axis=None),
                    p0=[
                        data.detuning[pair] + data.anharmonicity[pair],
                        data.flux_coefficient[pair],
                        0.07,
                        0,
                    ],
                    maxfev=100000,
                )
                fitted_parameters[pair].append(popt.tolist())
                duration[pair].append(np.pi / popt[2])
                half_duration[pair].append(np.pi / popt[2] / 2)
                amplitude[pair].append(
                    _resonance_amplitude(np.sqrt(-popt[0] / popt[1]), data.amplitude)
                )
            except Exception as e:
                fitted_parameters[pair].append([])
                duration[pair].append([])
                half_duration[pair].append([])
                amplitude[pair].append([])
                log.warning(f"Chevron fit failed for pair {pair} due to {e}")

    return ChevronResults(
        native=data.native,
        fitted_parameters=fitted_parameters,
        duration=duration,
        half_duration=half_duration,
        amplitude=amplitude,
    )


def _plot(data: ChevronData, fit: ChevronResults, target: QubitPairId):
    """Plot the experiment result for a single pair."""
    level2_qubit = data.level2_qubit.get(target)
    inverted = [" (inverted)" if qubit == level2_qubit else "" for qubit in target]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Qubit {target[0]} - Low Frequency - Data{inverted[0]}",
            f"Qubit {target[1]} - High Frequency - Data{inverted[1]}",
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
        names = [f"{fit.native} amplitude", f"{fit.native} duration"]
        values = [
            np.mean(fit.amplitude[target]),
            np.round(np.mean(fit.duration[target]))
            if data.native == "CZ"
            else np.round(np.mean(fit.half_duration[target])),
        ]
        if level2_qubit is not None:
            names.append("CZ interaction")
            values.append("|11>-|02>" if level2_qubit == target[1] else "|11>-|20>")
        fitting_report = table_html(table_dict(target[1], names, values))

    return [fig], fitting_report


def _update(
    results: ChevronResults, platform: CalibrationPlatform, target: QubitPairId
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


chevron = Protocol(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
