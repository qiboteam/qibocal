from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from qibocal import calibration, update
from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
)
from qibocal.protocols.ramsey.processing import MAXIMUM_FIT_POINTS
from qibocal.protocols.utils import table_dict, table_html

EPS = 1  # Hz
"""we add 1Hz when computing Delta frequency between the two qubit frequencies in order to avoid numerical error."""


@dataclass
class ZZInteractionParameters(Parameters):
    """Parameters for ZZ-interaction experiments."""

    delay_range: tuple[float, float, float]
    """delay time range (start, stop, step) in the sequence. Applied twice."""


ZZIntType = np.dtype(
    [
        ("delay", np.float64),
        ("targ_prob", np.float64),
        ("targ_error", np.float64),
        ("spect_prob", np.float64),
        ("spect_error", np.float64),
    ]
)
"""Custom dtype for ZZ-Interaction routines."""


@dataclass
class ZZInteractionData(Data):
    """Data for ZZ-interaction experiments."""

    qubit_freqs: dict[QubitId, float] = field(default_factory=dict)
    """Qubit freqs for each qubit."""
    anharmonicity: dict[QubitId, float] = field(default_factory=dict)
    """Targets qubit anharmonicity."""

    @property
    def delays(self) -> npt.NDArray:
        """
        Return a list with the delay times without repetitions.
        """
        qubit = next(iter(self.data))
        return np.unique(self.data[qubit].delay)


@dataclass
class ZZInteractionResults(Results):
    """Container for ZZ-interaction experiments results."""

    zz: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Estimated ZZ coupling of a qubit pair."""
    coupling: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Estimated coupling strenght of a qubit pair."""


def coupling_strength(
    omega1: float,
    omega2: float,
    anharmonicity1: float,
    anharmonicity2: float,
    zz: list[float],
) -> list[float]:
    """Compute the ZZ coupling from the difference in frequency and anharmonicity.

    coupling computing by inverting the following formula
    delta_q = omega1 - omega2
    xi = 2 g**2 (1 / (delta_q - alpha_2) - 1 / (delta_q + alpha_1))
    where delta_q is the difference in frequency and alpha_i is the anharmonicity
    """

    if anharmonicity1 == 0 or anharmonicity2 == 0:
        raise ValueError(
            "Anhamronicities are not estimated: cannot compute coupling strength."
        )

    # adding an eps to avoid numerical issues
    detuning = omega1 - omega2 + EPS
    denominator = 1 / (detuning - anharmonicity2) - 1 / (detuning + anharmonicity1)

    # here we compute coupling as a frequency and do error propagation
    return [
        float(np.sqrt(np.abs(zz[0] / 2 / denominator))),
        float(zz[1] / 2 / np.sqrt(2 * abs(denominator * zz[0]))),
    ]


def signal_plot(
    signal: npt.NDArray[ZZIntType],
    module: Callable,
    fit_params: list[float] | None = None,
    label: Literal["I", "X", ""] = "",
) -> tuple[list[go.Scatter], list[go.Scatter]]:
    """Plotting function for ZZ Interaction experiments."""

    target_scatters: list[go.Scatter] = []
    spectator_scatter: list[go.Scatter] = []

    delays = np.unique(signal.delay)

    signal_style = dict(color="red" if label == "I" else "blue")

    target_scatters.append(
        go.Scatter(
            x=delays,
            y=signal.targ_prob,
            error_y=dict(
                type="data",
                array=signal.targ_error,
                visible=True,
            ),
            opacity=1,
            name="Data " + label,
            showlegend=True,
            legendgroup="Data" + label,
            mode="markers",
            marker=signal_style,
        )
    )

    spectator_scatter.append(
        go.Scatter(
            x=delays,
            y=signal.spect_prob,
            error_y=dict(
                type="data",
                array=signal.spect_error,
                visible=True,
            ),
            opacity=1,
            name="Data " + label,
            showlegend=False,
            legendgroup="Data" + label,
            mode="markers",
            marker=signal_style,
        )
    )

    if fit_params is not None:
        fit_delays = np.linspace(min(delays), max(delays), MAXIMUM_FIT_POINTS)
        target_scatters.append(
            go.Scatter(
                x=fit_delays,
                y=module(fit_delays, *fit_params),
                name=f"Fit {label}",
                mode="lines",
                line=signal_style,
            ),
        )

    return target_scatters, spectator_scatter


def create_report_table(target: QubitPairId, fit: ZZInteractionResults):
    """Create an HTML table summarizing ZZ-interaction fit results for a qubit pair."""

    targ, spect = target
    fitting_report = table_html(
        table_dict(
            [targ] * 2,
            [
                f"ZZ  with {targ} [Hz]",
                f"Coupling with {spect} [Hz]",
            ],
            [
                fit.zz[target] if target in fit.zz else (0, 0),
                fit.coupling[target] if target in fit.coupling else (0, 0),
            ],
            display_error=True,
        ),
    )

    return fitting_report


def zz_update(
    results: ZZInteractionResults,
    platform: calibration.CalibrationPlatform,
    target: QubitPairId,
) -> None:
    """Update the platform calibration with the results of the Ramsey ZZ experiment."""
    if target in results.coupling:
        update.pair_coupling(results.coupling[target], platform, target)
