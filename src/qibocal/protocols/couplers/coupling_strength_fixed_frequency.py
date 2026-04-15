from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    ParallelSweepers,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Data, Parameters, QubitPairId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.ramsey import utils as ramsey_utils
from qibocal.protocols.utils import (
    HZ_TO_MHZ,
    table_dict,
    table_html,
)

__all__ = ["tunable_coupling_fixed_frequency"]


class InvalidInputParameters(Exception):
    pass


class Setup(str, Enum):
    Id = "Id"
    X = "X"


@dataclass
class TunableCouplingParameters(Parameters):
    """TunableCoupling runcard inputs."""

    delay_between_pulses_range: tuple[int, int, int] | None = None
    """Delay range between the two RX(pi/2) pulses in ns."""
    delay_between_pulses_start: int | None = None
    """Initial delay between RX(pi/2) pulses in ns.property"""
    delay_between_pulses_end: int | None = None
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int | None = None
    """Step delay between RX(pi/2) pulses in ns."""
    flux_ampl_range: tuple[float, float, float] | None = None
    """Coupler flux DC offset range. Same for all qbuits"""
    flux_ampl_start: float | None = None
    """Initial coupler flux DC offset. Same for all qubits."""
    flux_ampl_end: float | None = None
    """Final coupler flux DC offset. Same for all qubits."""
    flux_ampl_step: float | None = None
    """Step coupler flux DC offset. Same for all qubits."""
    detuning: int | None = None
    """Frequency detuning [Hz] (optional).
    If 0 standard Ramsey experiment is performed."""
    verbose_plot: bool | None = False

    @property
    def duration_range(self) -> tuple[int, int, int]:

        if self.delay_between_pulses_range is None:
            return (
                self.delay_between_pulses_start,
                self.delay_between_pulses_end,
                self.delay_between_pulses_step,
            )
        return self.delay_between_pulses_range

    @property
    def flux_range(self) -> tuple[float, float, float]:
        if self.flux_ampl_range is None:
            return (
                self.flux_ampl_start,
                self.flux_ampl_end,
                self.flux_ampl_step,
            )
        return self.flux_ampl_range

    def __post_init__(self):
        if (
            self.duration_range is None
            or any([d < 0 for d in self.duration_range])
            or self.duration_range[0] >= self.duration_range[1]
        ):
            raise InvalidInputParameters("Pulse delays not set properly.")
        if self.flux_range is None:
            raise InvalidInputParameters("Flux offset not set properly.")

    def duration_values(self) -> npt.NDArray:
        return np.arange(*self.duration_range)

    def flux_values(self) -> npt.NDArray:
        return np.arange(*self.flux_ampl_range)


class ZZ(TypedDict):
    interaction: float
    ramsey_data: dict[Setup, dict[str, list[float]]]


@dataclass
class TunableCouplingResults(Results):
    """TunableCoupling outputs."""

    detuning: float
    """Qubit frequency detuning."""
    zz: dict[QubitPairId, dict[float, ZZ]] = field(default_factory=dict)
    "zz interaction strength for different offsets values"
    min_interaction_point: dict[QubitPairId, tuple[float, float]] = field(
        default_factory=tuple
    )


TunableCouplingType = np.dtype(
    [("wait", np.int64), ("offset", np.float64), ("prob", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class TunableCouplingData(Data):
    """TunableCoupling acquisition outputs."""

    qubit_pairs: list[QubitPairId]
    flux_offsets: list[float]
    detuning: Optional[int] = None
    """Frequency detuning [Hz]."""
    pairs_freqs: dict[QubitPairId, tuple[float, float]] = field(default_factory=dict)
    """Frequencies of pairs of qubit."""
    pairs_anharmonicities: dict[QubitPairId, tuple[float, float]] = field(
        default_factory=dict
    )
    """Anharmonicity of pairs of qubit."""
    data: dict[tuple[QubitPairId, Setup], npt.NDArray[TunableCouplingType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""
    verbose_plot: Optional[bool] = False

    def register_qubit(self, dtype, data_keys, data_dict):
        """Store output for single qubit."""
        delay_list = data_dict["wait"]
        offset_list = data_dict["offset"]
        size = len(delay_list) * len(offset_list)
        ar = np.empty(size, dtype=dtype)
        offsets, delays = np.meshgrid(offset_list, delay_list)
        ar["wait"] = delays.ravel()
        ar["offset"] = offsets.ravel()
        ar["prob"] = data_dict["prob"].ravel()

        self.data[data_keys] = np.rec.array(ar)


def _acquisition(
    params: TunableCouplingParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> TunableCouplingData:
    """Acquire tunable coupling strength data."""

    data = TunableCouplingData(
        qubit_pairs=targets,
        flux_offsets=params.flux_values().tolist(),
        detuning=params.detuning,
        pairs_freqs={
            pair: (
                platform.config(platform.qubits[pair[0]].drive).frequency,
                platform.config(platform.qubits[pair[1]].drive).frequency,
            )
            for pair in targets
        },
        pairs_anharmonicities={
            pair: (
                platform.calibration.single_qubits[pair[0]].qubit.anharmonicity,
                platform.calibration.single_qubits[pair[1]].qubit.anharmonicity,
            )
            for pair in targets
        },
        verbose_plot=params.verbose_plot,
    )

    coupler_map = platform.find_coupler_for_pair_list(targets)

    targets = np.array(targets)
    for q1_state in Setup:
        ramsey_coupler_sequence = PulseSequence()

        qubit_delays = []
        coupler_pulses = []
        acquisition_delays = []
        for (q0, q1), coupler in zip(targets, coupler_map.values()):
            sequence = PulseSequence()
            if q1_state == Setup.X:
                assert q1 not in targets[:, 0], (
                    f"Cannot run Ramsey experiment on qubit {q1} if it is already in Ramsey sequence."
                )
                natives = platform.natives.single_qubit[q1]
                sequence.append(natives.RX()[0])

            q_rx_ch, q_rx_pulse = platform.natives.single_qubit[q0].RX()[0]
            q_acq_ch, q_acq_pulse = platform.natives.single_qubit[q0].MZ()[0]
            coupler_ch = platform.couplers[coupler].flux

            sequence.append((q_rx_ch, q_rx_pulse))

            qubit_delays.append(Delay(duration=params.duration_range[1]))
            acquisition_delays.append(Delay(duration=params.duration_range[1]))
            coupler_pulses.append(
                Pulse(
                    duration=params.duration_range[1],
                    amplitude=params.flux_range[1],
                    envelope=Rectangular(),
                )
            )

            sequence |= (
                (q_rx_ch, qubit_delays[-1]),
                (coupler_ch, coupler_pulses[-1]),
                (q_acq_ch, acquisition_delays[-1]),
            )

            sequence += (
                (q_rx_ch, q_rx_pulse),
                (q_acq_ch, Delay(duration=q_rx_pulse.duration)),
                (q_acq_ch, q_acq_pulse),
            )

        ramsey_coupler_sequence += sequence

        coupler_flux_parsweepers = ParallelSweepers(
            [
                Sweeper(
                    parameter=Parameter.amplitude,
                    range=params.flux_range,
                    pulses=coupler_pulses,
                )
            ]
        )

        time_parsweepers = ParallelSweepers(
            [
                Sweeper(
                    parameter=Parameter.duration,
                    range=params.duration_range,
                    pulses=coupler_pulses + acquisition_delays + qubit_delays,
                ),
            ]
        )

        results = platform.execute(
            [sequence],
            [time_parsweepers, coupler_flux_parsweepers],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        for q in targets.tolist():
            ro_pulse = list(sequence.channel(platform.qubits[q[0]].acquisition))[-1]

            result = results[ro_pulse.id]
            data.register_qubit(
                TunableCouplingType,
                (tuple(q), q1_state),
                dict(
                    wait=params.duration_values(),
                    offset=params.offset_values(),
                    prob=result,
                ),
            )

    return data


def _fit(data: TunableCouplingData) -> TunableCouplingResults:
    """Post-processing for TunableCoupling experiment."""

    zz = {}
    min_interaction = {}
    flux_offsets = np.array(data.flux_offsets)
    for pair in data.qubit_pairs:
        pair = tuple(pair)

        q0_freq, _ = data.pairs_freqs[pair]

        try:
            zz_off = {}

            for offset in flux_offsets:
                ramsey_dict = {}

                if all(
                    [
                        np.std(
                            data.data[pair, setup].prob[
                                data.data[pair, setup].offset == offset
                            ]
                        )
                        < 0.05
                        for setup in Setup
                    ]
                ):
                    zz_off[offset] = {
                        "ramsey_data": {},
                        "interaction": 0.0,
                    }

                else:
                    for setup in Setup:
                        pair_data = data.data[pair, setup]
                        waits = pair_data.wait[pair_data.offset == offset]
                        probs = pair_data.prob[pair_data.offset == offset]

                        popt, perr = ramsey_utils.fitting(waits, probs)
                        perr = [p if np.isfinite(p) else 0 for p in perr]
                        fit_params = ramsey_utils.process_fit(
                            popt, perr, q0_freq, data.detuning
                        )

                        ramsey_dict[setup] = {
                            "frequency": fit_params[0],
                            "t2": fit_params[1],
                            "delta_phys": fit_params[2],
                            "delta_fitting": fit_params[3],
                            "fitted_parameters": fit_params[4],
                        }

                    zz_off[offset] = {
                        "ramsey_data": ramsey_dict,
                        "interaction": float(
                            ramsey_dict[Setup.X]["frequency"][0]
                            - ramsey_dict[Setup.Id]["frequency"][0]
                        ),
                    }

            zz[pair] = zz_off

            zz_vs_offset = [zz[pair][f]["interaction"] for f in flux_offsets]
            min_zz = np.min(np.abs(zz_vs_offset))
            min_zz_indices = np.where(np.abs(zz_vs_offset) == min_zz)[0]
            selected_offset = np.min(flux_offsets[min_zz_indices])
            selected_offset_index = np.where(
                np.abs(flux_offsets) == np.abs(selected_offset)
            )[0][0]
            min_interaction[pair] = (
                selected_offset,
                zz_vs_offset[selected_offset_index],
            )

        except Exception as e:
            log.warning(
                f"Coupling Strength fitting failed for qubit pair {pair} due to {e}."
            )

    return TunableCouplingResults(
        detuning=data.detuning,
        zz=zz,
        min_interaction_point=min_interaction,
    )


def _plot(
    data: TunableCouplingData,
    fit: Optional[TunableCouplingResults],
    target: QubitPairId,
):
    """Plotting function for TunableCoupling experiment."""

    figures = []
    fitting_report = ""

    for flux in data.flux_offsets:
        ramsey_fig = go.Figure()
        for setup in Setup:
            selected_data = data.data[target, setup][
                data.data[target, setup].offset == flux
            ]
            ramsey_fig.add_trace(
                go.Scatter(
                    x=selected_data.wait,
                    y=selected_data.prob,
                    name=f"Control: {setup} - coupler flux: {np.round(flux, 2)}",
                    mode="markers",
                ),
            )
            if fit is not None and fit.zz[target][flux]["ramsey_data"]:
                fit_waits = np.linspace(
                    np.min(selected_data.wait), np.max(selected_data.wait), 200
                )
                ramsey_fig.add_trace(
                    go.Scatter(
                        x=fit_waits,
                        y=ramsey_utils.ramsey_fit(
                            fit_waits,
                            *fit.zz[target][flux]["ramsey_data"][setup][
                                "fitted_parameters"
                            ],
                        ),
                        name=f"Fit Control: {setup} - coupler flux: {np.round(flux, 2)}",
                        line=go.scatter.Line(dash="dashdot"),
                    ),
                )

        figures.append(ramsey_fig)

    if fit is not None:
        interaction_fig = go.Figure()

        interaction_fig.add_trace(
            go.Scatter(
                x=list(fit.zz[target].keys()),
                y=np.array([v["interaction"] for v in fit.zz[target].values()])
                * HZ_TO_MHZ,
                name="zz interaction",
                mode="lines+markers",
                line=go.scatter.Line(dash="dot"),
            ),
        )

        interaction_fig.add_trace(
            go.Scatter(
                x=[fit.min_interaction_point[target][0]],
                y=[fit.min_interaction_point[target][1] * HZ_TO_MHZ],
                name="minimum zz",
                line=go.scatter.Line(dash="dot"),
            ),
        )

        interaction_fig.update_layout(
            showlegend=True,
            xaxis_title="Flux Bias [V]",
            yaxis_title="ZZ [MHz]",
            title="ZZ Interaction vs Coupler Frequency",
        )

        fitting_report = table_html(
            table_dict(
                [target],
                [
                    "Min ZZ interaction [MHz]",
                ],
                [
                    np.round(fit.min_interaction_point[target][1] * 1e-3, 3),
                ],
            )
        )

        figures.append(interaction_fig)

    return figures, fitting_report


def _update(
    results: TunableCouplingResults, platform: CalibrationPlatform, target: QubitPairId
):
    pass


tunable_coupling_fixed_frequency = Routine(_acquisition, _fit, _plot, _update)
"""TunableCoupling Routine object."""
