from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Readout, Sweeper

from qibocal.calibration import CalibrationPlatform

from ...auto.operation import QubitId, Routine
from ...config import log
from ...result import probability
from ..utils import table_dict, table_html
from .ramsey import (
    COLORBAND,
    COLORBAND_LINE,
    RamseySignalData,
    RamseySignalParameters,
    RamseySignalResults,
    RamseyType,
    _update,
)
from .utils import fitting, process_fit, ramsey_fit, ramsey_sequence


@dataclass
class RamseyZZParameters(RamseySignalParameters):
    """RamseyZZ runcard inputs."""

    target_qubit: Optional[QubitId] = None
    """Target qubit that will be excited."""


@dataclass
class RamseyZZResults(RamseySignalResults):
    """RamseyZZ outputs."""

    def __contains__(self, qubit: QubitId):
        # TODO: to be improved
        return all(
            list(getattr(self, field.name))[0][0] == qubit
            for field in fields(self)
            if isinstance(getattr(self, field.name), dict)
        )


@dataclass
class RamseyZZData(RamseySignalData):
    """RamseyZZ acquisition outputs."""

    target_qubit: Optional[QubitId] = None
    """Qubit that will be excited."""
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
        target_qubit=params.target_qubit,
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
                target_qubit=params.target_qubit if setup == "X" else None,
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
                    target_qubit=params.target_qubit if setup == "X" else None,
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
    return RamseyZZResults(
        detuning=data.detuning,
        frequency=freq_measure,
        t2=t2_measure,
        delta_phys=delta_phys_measure,
        delta_fitting=delta_fitting_measure,
        fitted_parameters=popts,
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
                data.target_qubit,
                [
                    "ZZ  [kHz]",
                ],
                [
                    np.round(
                        (fit.frequency[target, "X"][0] - fit.frequency[target, "I"][0])
                        * 1e-3,
                        0,
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
