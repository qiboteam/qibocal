from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Readout, Sweeper

from qibocal.calibration import CalibrationPlatform

from ...auto.operation import QubitId, Routine
from ...config import log
from ...result import probability
from ..utils import COLORBAND, COLORBAND_LINE, table_dict, table_html
from .ramsey import RamseyType
from .ramsey_signal import (
    RamseySignalData,
    RamseySignalParameters,
    RamseySignalResults,
    _update,
)
from .utils import fitting, process_fit, ramsey_fit, ramsey_sequence

__all__ = ["ramsey_zz"]

EPS = 1  # Hz


def compute_coupling_strength(
    omega1: Union[npt.NDArray, float],
    omega2: Union[npt.NDArray, float],
    anharmonicity1: Union[npt.NDArray, float],
    anharmonicity2: Union[npt.NDArray, float],
    zz: Union[npt.NDArray, float],
) -> npt.NDArray:
    """Compute the ZZ coupling from the difference in frequency and anharmonicity.

    coupling computing by inverting the following formula
    delta_q = omega1 - omega2
    xi = 2 g**2 (1 / (delta_q - alpha_2) - 1 / (delta_q + alpha_1))
    where delta_q is the difference in frequency and alpha_i is the anharmonicity
    """
    # adding an eps to avoid numerical issues
    delta_qubit_freq = omega1 - omega2 + EPS
    denominator = 1 / (delta_qubit_freq - anharmonicity2) - 1 / (
        delta_qubit_freq + anharmonicity1
    )

    # here we compute coupling as a frequency
    return np.sqrt(np.abs(zz / 2 / denominator))


@dataclass
class RamseyZZParameters(RamseySignalParameters):
    """RamseyZZ runcard inputs."""

    spectator_qubit: Optional[QubitId] = None
    """Spectator qubit that will be excited."""


@dataclass
class RamseyZZResults(RamseySignalResults):
    """RamseyZZ outputs."""

    zz: dict[QubitId, float] = field(default_factory=dict)
    coupling: dict[QubitId, float] = field(default_factory=dict)

    def __contains__(self, qubit: QubitId):
        return qubit in self.zz


@dataclass
class RamseyZZData(RamseySignalData):
    """RamseyZZ acquisition outputs."""

    spectator_qubit: Optional[QubitId] = None
    """Qubit that will be excited."""
    anharmonicity: dict[QubitId, float] = field(default_factory=dict)
    """Targets qubit anharmonicity."""
    anharmonicity_spectator_qubit: float = None
    """Anharmonicity of spectator qubit."""
    frequency_spectator_qubit: float = 0
    "Frequency of spectator qubit."
    data: dict[tuple[QubitId, str], npt.NDArray[RamseyType]] = field(
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

    waits = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

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
        anharmonicity_spectator_qubit=platform.calibration.single_qubits[
            params.spectator_qubit
        ].qubit.anharmonicity,
        frequency_spectator_qubit=platform.config(
            platform.qubits[params.spectator_qubit].drive
        ).frequency,
        spectator_qubit=params.spectator_qubit,
    )

    updates = []
    if params.detuning is not None:
        for qubit in targets:
            channel = platform.qubits[qubit].drive
            f0 = platform.config(channel).frequency
            updates.append({channel: {"frequency": f0 + params.detuning}})

    for setup in ["I", "X"]:
        if not params.unrolling:
            sequence, delays = ramsey_sequence(
                platform=platform,
                targets=targets,
                spectator_qubit=params.spectator_qubit if setup == "X" else None,
            )

            sweeper = Sweeper(
                parameter=Parameter.duration,
                values=waits,
                pulses=delays,
            )

            # execute the sweep
            results = platform.execute(
                [sequence],
                [[sweeper]],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.SINGLESHOT,
                updates=updates,
            )

            for qubit in targets:
                ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[
                    -1
                ]
                probs = probability(results[ro_pulse.id], state=1)
                errors = [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs]

        else:
            sequences, all_ro_pulses = [], []
            probs, errors = [], []
            for wait in waits:
                sequence, _ = ramsey_sequence(
                    platform=platform,
                    targets=targets,
                    wait=wait,
                    spectator_qubit=params.spectator_qubit if setup == "X" else None,
                )
                sequences.append(sequence)
                all_ro_pulses.append(
                    {
                        qubit: [
                            readout
                            for readout in sequence.channel(
                                platform.qubits[qubit].acquisition
                            )
                            if isinstance(readout, Readout)
                        ][0]
                        for qubit in targets
                    }
                )

            results = platform.execute(
                sequences,
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.SINGLESHOT,
                updates=updates,
            )

            for wait, ro_pulses in zip(waits, all_ro_pulses):
                for qubit in targets:
                    result = results[ro_pulses[qubit].id]
                    prob = probability(result, state=1)
                    probs.append(prob)
                    errors.append(np.sqrt(prob * (1 - prob) / params.nshots))

        for qubit in targets:
            data.register_qubit(
                RamseyType,
                (qubit, setup),
                dict(
                    wait=waits,
                    prob=probs,
                    errors=errors,
                ),
            )

    return data


def _fit(data: RamseyZZData) -> RamseyZZResults:
    """Fitting procedure for RamseyZZ protocol.

    Standard Ramsey fitting procedure is applied for both version of
    the experiment.

    """
    waits = data.waits
    popts = {}
    freq_measure = {}
    t2_measure = {}
    delta_phys_measure = {}
    delta_fitting_measure = {}
    zz = {}
    coupling = {}
    for qubit in data.qubits:
        for setup in ["I", "X"]:
            qubit_data = data[qubit, setup]
            qubit_freq = data.qubit_freqs[qubit]
            probs = qubit_data["prob"]
            try:
                popt, perr = fitting(waits, probs, qubit_data.errors)
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
        zz[qubit] = float(freq_measure[qubit, "X"][0] - freq_measure[qubit, "I"][0])

        # here we compute coupling as a frequency
        coupling[qubit] = float(
            compute_coupling_strength(
                omega1=data.qubit_freqs[qubit],
                omega2=data.frequency_spectator_qubit,
                anharmonicity1=data.anharmonicity[qubit],
                anharmonicity2=data.anharmonicity_spectator_qubit,
                zz=zz[qubit],
            )
        )

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


def _plot(data: RamseyZZData, target: QubitId, fit: RamseyZZResults = None):
    """Plotting function for Ramsey Experiment."""
    figures = []
    fitting_report = ""

    waits = data[target, "I"].wait
    probs_I = data.data[target, "I"].prob
    probs_X = data.data[target, "X"].prob

    error_bars_I = data.data[target, "I"].errors
    error_bars_X = data.data[target, "X"].errors
    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=probs_I,
                opacity=1,
                name="I",
                showlegend=True,
                legendgroup="I ",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((waits, waits[::-1])),
                y=np.concatenate(
                    (probs_I + error_bars_I, (probs_I - error_bars_I)[::-1])
                ),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors I",
            ),
            go.Scatter(
                x=waits,
                y=probs_X,
                opacity=1,
                name="X",
                showlegend=True,
                legendgroup="X",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((waits, waits[::-1])),
                y=np.concatenate(
                    (probs_X + error_bars_X, (probs_X - error_bars_X)[::-1])
                ),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors X",
            ),
        ]
    )

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=waits,
                y=ramsey_fit(waits, *fit.fitted_parameters[target, "I"]),
                name="Fit I",
                line=go.scatter.Line(dash="dot"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=waits,
                y=ramsey_fit(waits, *fit.fitted_parameters[target, "X"]),
                name="Fit X",
                line=go.scatter.Line(dash="dot"),
            )
        )
        fitting_report = table_html(
            table_dict(
                target,
                [
                    f"ZZ  with {data.spectator_qubit} [kHz]",
                    f"Coupling with {data.spectator_qubit} [MHz]",
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

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Excited state probability",
    )

    figures.append(fig)

    return figures, fitting_report


ramsey_zz = Routine(_acquisition, _fit, _plot, _update)
"""Ramsey Routine object."""
