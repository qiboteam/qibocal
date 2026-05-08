from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ..utils import table_dict, table_html
from .acquisition import (
    RamseyData,
    RamseyParameters,
    RamseyResults,
    execute_experiment,
    ramsey_sequence,
)
from .processing import (
    MAXIMUM_FIT_POINTS,
    fitting,
    process_fit,
    ramsey_fit,
    ramsey_update,
)

__all__ = ["ramsey_zz"]


@dataclass
class RamseyZZParameters(RamseyParameters):
    """RamseyZZ runcard inputs."""

    target_qubit: QubitId | None = None
    """Target qubit that will be excited."""


@dataclass
class RamseyZZResults(RamseyResults):
    """RamseyZZ outputs."""

    zz: dict[QubitId, float] = field(default_factory=dict)
    coupling: dict[QubitId, float] = field(default_factory=dict)

    def __contains__(self, qubit: QubitId):
        return qubit in self.zz


RamseyZZType = np.dtype([("wait", np.float64), ("prob", np.float64)])
"""Custom dtype for coherence routines."""


@dataclass
class RamseyZZData(RamseyData):
    """RamseyZZ acquisition outputs."""

    target_qubit: QubitId | None = None
    """Qubit that will be excited."""
    anharmonicity: dict[QubitId, float] = field(default_factory=dict)
    """Targets qubit anharmonicity."""
    anharmonicity_target_qubit: float = None
    """Anharmonicity of target qubit."""
    frequency_target_qubit: float = 0
    "Frequency of target qubit."
    data: dict[tuple[QubitId, str], npt.NDArray[RamseyZZType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _acquisition(
    params: RamseyZZParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RamseyZZData:
    """Data acquisition for RamseyZZ Experiment.

    Standard Ramsey experiment repeated twice.
    In the second execution one qubit is brought to the excited state.
    """

    data = RamseyZZData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.config(platform.qubits[qubit].drive).frequency
            for qubit in targets
        },
        anharmonicity={
            qubit: platform.calibration.single_qubits[qubit].qubit.anharmonicity
            for qubit in targets
        },
        anharmonicity_target_qubit=platform.calibration.single_qubits[
            params.target_qubit
        ].qubit.anharmonicity,
        frequency_target_qubit=platform.config(
            platform.qubits[params.target_qubit].drive
        ).frequency,
        target_qubit=params.target_qubit,
    )

    for setup in ["I", "X"]:
        sequence, delays = ramsey_sequence(
            platform=platform,
            targets=targets,
            target_qubit=params.target_qubit if setup == "X" else None,
        )

        results = execute_experiment(
            sequence=sequence,
            delays=delays,
            platform=platform,
            targets=targets,
            params=params,
            return_probs=True,
        )

        for qubit in targets:
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]

            data.register_qubit(
                RamseyZZType,
                (qubit, setup),
                dict(
                    wait=np.arange(*params.delay_range),
                    prob=results[ro_pulse.id],
                ),
            )

    return data


def _fit(data: RamseyZZData) -> RamseyZZResults:
    """Fitting procedure for RamseyZZ protocol.

    Standard Ramsey fitting procedure is applied for both version of
    the experiment.

    """
    waits = data.waits
    popts: dict[QubitId, list[float]] = {}
    freq_measure: dict[QubitId, list[float]] = {}
    t2_measure: dict[QubitId, list[float]] = {}
    delta_phys_measure: dict[QubitId, list[float]] = {}
    delta_fitting_measure: dict[QubitId, list[float]] = {}
    zz: dict[QubitId, float] = {}
    coupling: dict[QubitId, float] = {}
    for qubit in data.qubits:
        for setup in ["I", "X"]:
            qubit_data = data[qubit, setup]
            qubit_freq = data.qubit_freqs[qubit]
            probs = qubit_data["prob"]
            try:
                popt, perr = fitting(waits, probs)
                (
                    freq_measure[qubit, setup],
                    t2_measure[qubit, setup],
                    delta_phys_measure[qubit, setup],
                    delta_fitting_measure[qubit, setup],
                    popts[qubit, setup],
                ) = process_fit(popt, perr, qubit_freq, data.detuning)
            except Exception as e:
                log.warning(f"Ramsey fitting failed for qubit {qubit} due to {e}.")
        # compute zz and qq coupling
        # zz the difference in frequency between the two measurement
        zz[qubit] = float(
            np.abs(freq_measure[qubit, "X"][0] - freq_measure[qubit, "I"][0])
        )

        # coupling computing by inverting the following formula
        # xi = 2 g**2 (1 / (delta_q - alpha_1) - 1 / (delta_q + alpha_2))
        # where delta_q is the difference in frequency and alpha_i is the anharmonicity
        delta_qubit_freq = np.abs(data.qubit_freqs[qubit] - data.frequency_target_qubit)
        denominator = -(
            1 / (delta_qubit_freq - data.anharmonicity[qubit])
            - 1 / (delta_qubit_freq + data.anharmonicity_target_qubit)
        )
        coupling[qubit] = float(np.sqrt(zz[qubit] / 2 / np.abs(denominator)))
    return RamseyZZResults(
        detuning=data.detuning,
        frequency=freq_measure,
        t2=t2_measure,
        delta_phys=delta_phys_measure,
        delta_fitting=delta_fitting_measure,
        fitted_parameters=popts,
        zz=zz,
        coupling=coupling,
    )


def zz_fit_plot(
    target: QubitId,
    spect_qubit: QubitId,
    fit: RamseyZZResults,
    waits: npt.NDArray,
    fig: go.Figure,
) -> str:
    fit_waits = np.linspace(min(waits), max(waits), MAXIMUM_FIT_POINTS)

    fig.add_trace(
        go.Scatter(
            x=fit_waits,
            y=ramsey_fit(fit_waits, *fit.fitted_parameters[target, "I"]),
            name="Fit I",
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fit_waits,
            y=ramsey_fit(fit_waits, *fit.fitted_parameters[target, "X"]),
            name="Fit X",
            mode="lines",
        )
    )
    fitting_report = table_html(
        table_dict(
            target,
            [
                f"ZZ  with {spect_qubit} [kHz]",
                f"Coupling with {spect_qubit} [MHz]",
            ],
            [
                np.round(
                    fit.zz[target] * 1e-3,
                    0,
                ),
                np.round(
                    fit.coupling[target] * 1e-6,
                    2,
                ),
            ],
        )
    )

    return fitting_report


def _plot(
    data: RamseyZZData, target: QubitId, fit: RamseyZZResults | None = None
) -> tuple[list[go.Figure], str]:
    """Plotting function for Ramsey Experiment."""

    fitting_report = ""

    waits = data.waits
    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=data.data[target, "I"].prob,
                opacity=1,
                name="I",
                showlegend=True,
                legendgroup="I ",
                mode="markers",
            ),
            go.Scatter(
                x=waits,
                y=data.data[target, "X"].prob,
                opacity=1,
                name="X",
                showlegend=True,
                legendgroup="X",
                mode="markers",
            ),
        ]
    )

    if fit is not None:
        fitting_report = zz_fit_plot(
            target=target,
            spect_qubit=data.target_qubit,
            fit=fit,
            waits=waits,
            fig=fig,
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Excited state probability",
    )

    return [fig], fitting_report


ramsey_zz = Routine(_acquisition, _fit, _plot, ramsey_update)
"""Ramsey ZZ Routine object.

This protocol measures the state-dependent frequency shift (ZZ interaction)
between a selected target qubit and one spectator qubit. It
performs two Ramsey experiments for each qubit in the target list:

- "I" setup: spectator qubit remain in the ground state.
- "X" setup: spectator qubit is excited before the Ramsey sequence.

The fitted Ramsey frequencies from these two experiments are compared to
extract the conditional ZZ shift experienced by each measured qubit. The
difference between the two fitted frequencies is reported as the
ZZ interaction strength. Using the measured qubit frequencies, the target
qubit frequency, and the anharmonicities, the routine also estimates the
effective coupling strength between each qubit and the target qubit.
This protocol is useful for characterizing residual static coupling and
frequency shifts induced by an excited neighboring qubit. The plot output
shows Ramsey traces in probability for both the I and X setups.
"""
