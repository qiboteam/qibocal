"""CZ virtual correction experiment for two qubit gates, tune landscape."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platform import Platform

from qibocal.auto.operation import Data, Parameters, QubitsPairs, Results, Routine

from .cz_virtualz import cz_virtualz

COLORAXIS = ["coloraxis2", "coloraxis1"]


@dataclass
class TuneLandscapeSNZParameters(Parameters):
    """CzVirtualZ runcard inputs."""

    amplitude_min: float
    """Amplitude minimum."""
    amplitude_max: float
    """Amplitude maximum."""
    amplitude_step: float
    """Amplitude step."""
    duration: float
    """Pulse duration."""
    wait_min: float
    """Duration minimum."""
    wait_max: float
    """Duration maximum."""
    wait_step: float
    """Duration step."""
    dt: Optional[float] = 20
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""


@dataclass
class TuneLandscapeResults(Results):
    """CzVirtualZ outputs when fitting will be done."""


TuneLandscapeType = np.dtype(
    [
        ("leakage", np.float64),
        ("cz_angle", np.float64),
        ("phase", np.float64),
        ("length", np.float64),
        ("amp", np.float64),
    ]
)


@dataclass
class TuneLandscapeData(Data):
    """CZVirtualZ data."""

    data: dict[tuple, npt.NDArray[TuneLandscapeType]] = field(default_factory=dict)

    def __getitem__(self, pair):
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }

    def register_qubit(
        self, target_q, control_q, duration, amplitude, leakage, cz_angle, phase
    ):
        """Store output for single qubit."""
        ar = np.empty((1,), dtype=TuneLandscapeType)
        ar["length"] = duration
        ar["amp"] = amplitude
        ar["leakage"] = leakage
        ar["cz_angle"] = cz_angle
        ar["phase"] = phase

        if (target_q, control_q) in self.data:
            self.data[target_q, control_q] = np.rec.array(
                np.concatenate((self.data[target_q, control_q], ar))
            )
        else:
            self.data[target_q, control_q] = np.rec.array(ar)


# TODO: make acquisition working regardless of the qubits order
def _acquisition(
    params: TuneLandscapeSNZParameters,
    platform: Platform,
    qubits: QubitsPairs,
) -> TuneLandscapeData:
    r"""
    Acquisition for tune landscape.

    Currently this experiments perform the CZVirtualZ protocols for different
    amplitudes and durations of the flux pulse. The leakage and the CZ angle are
    plotted for each configuration.
    """

    data = TuneLandscapeData()
    amplitude_range = np.arange(
        params.amplitude_min,
        params.amplitude_max,
        params.amplitude_step,
    )

    wait_range = np.arange(params.wait_min, params.wait_max, params.wait_step)

    for amplitude in amplitude_range:
        for wait in wait_range:
            cz_data, _ = cz_virtualz.acquisition(
                params=cz_virtualz.parameters_type.load(
                    dict(
                        theta_start=0,
                        theta_end=7,
                        theta_step=0.1,
                        flux_pulse_amplitude=amplitude,
                        flux_pulse_duration=params.duration,
                        idling_time=wait,
                    )
                ),
                platform=platform,
                qubits=qubits,
            )
            cz_fit, _ = cz_virtualz.fit(cz_data)

            for pair in qubits:
                for target_q, control_q in (
                    pair,
                    list(pair)[::-1],
                ):
                    data.register_qubit(
                        target_q,
                        control_q,
                        wait,
                        amplitude,
                        leakage=cz_fit.leakage[pair][control_q],
                        cz_angle=cz_fit.cz_angle[pair],
                        phase=cz_fit.virtual_phase[pair][target_q],
                    )

    return data


def _fit(
    data: TuneLandscapeData,
) -> TuneLandscapeResults:
    return TuneLandscapeResults()


def _plot(data: TuneLandscapeData, fit: TuneLandscapeResults, qubit):
    pair_data = data[qubit]
    qubits = next(iter(pair_data))[:2]
    fig1 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Leakage Qubit {qubits[0]}",
            f"Leakage Qubit {qubits[1]}",
        ),
    )
    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"CZ angle Qubit {qubits[0]}",
            f"CZ angle Qubit {qubits[1]}",
        ),
    )

    for target, control in pair_data:
        fig1.add_trace(
            go.Heatmap(
                x=pair_data[target, control].length,
                y=pair_data[target, control].amp,
                z=abs(
                    pair_data[target, control].leakage
                ),  # TODO: check if you need abs
                name=f"Leakage qubit {target}",
                colorbar=dict(
                    tickmode="array",
                    tickvals=[0, 0.1, 0.2, 0.3, 0.4],
                    ticktext=["0", "0.1", "0.2", "0.3", "0.4"],  # Set the tick labels
                ),
            ),
            row=1,
            col=1 if (target, control) == qubits else 2,
        )

        fig2.add_trace(
            go.Heatmap(
                x=pair_data[target, control].length,
                y=pair_data[target, control].amp,
                z=pair_data[target, control].cz_angle,
                colorscale="RdBu",
                colorbar=dict(
                    tickmode="array",
                    tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                    ticktext=[
                        "0",
                        "1.57",
                        "3.14",
                        "4.71",
                        "6.28",
                    ],  # Set the tick labels
                ),
                name=f"CZ angle [rad] qubit {target}",
            ),
            row=1,
            col=1 if (target, control) == qubits else 2,
        )
        fig1.update_layout(
            showlegend=True,
            xaxis1_title="Idling time",
            xaxis2_title="Idling time",
            yaxis_title="Flux pulse amplitude",
        )

        fig2.update_layout(
            showlegend=True,
            xaxis1_title="Idling time",
            xaxis2_title="Idling time",
            yaxis_title="Flux pulse amplitude",
        )

    return [fig1, fig2], ""


# def _update(results: CZVirtualZResults, platform: Platform, qubit_pair: QubitPairId):
#     # FIXME: quick fix for qubit order
#     qubit_pair = tuple(sorted(qubit_pair))
#     update.virtual_phases(results.virtual_phase[qubit_pair], platform, qubit_pair)


tune_landscape_snz = Routine(_acquisition, _fit, _plot, two_qubit_gates=True)
"""CZ virtual Z correction routine."""
