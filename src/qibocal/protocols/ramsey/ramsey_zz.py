from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from ...auto.operation import Routine
from ...config import log
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
    platform: Platform,
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

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    data = RamseyZZData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.qubits[qubit].native_gates.RX.frequency for qubit in targets
        },
        target_qubit=params.target_qubit,
    )

    for setup in ["I", "X"]:
        if not params.unrolling:
            sequence = PulseSequence()
            for qubit in targets:
                sequence += ramsey_sequence(
                    platform=platform,
                    qubit=qubit,
                    detuning=params.detuning,
                    target_qubit=params.target_qubit if setup == "X" else None,
                )

            sweeper = Sweeper(
                Parameter.start,
                waits,
                [sequence.get_qubit_pulses(qubit).qd_pulses[-1] for qubit in targets],
                type=SweeperType.ABSOLUTE,
            )

            # execute the sweep
            results = platform.sweep(
                sequence,
                options,
                sweeper,
            )

            for qubit in targets:
                probs = results[qubit].probability(state=1)
                errors = [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs]

        else:
            sequences, all_ro_pulses = [], []
            probs, errors = [], []
            for wait in waits:
                sequence = PulseSequence()
                for qubit in targets:
                    sequence += ramsey_sequence(
                        platform=platform,
                        qubit=qubit,
                        wait=wait,
                        detuning=params.detuning,
                        target_qubit=params.target_qubit if setup == "X" else None,
                    )

                sequences.append(sequence)
                all_ro_pulses.append(sequence.ro_pulses)

            results = platform.execute_pulse_sequences(sequences, options)

            for wait, ro_pulses in zip(waits, all_ro_pulses):
                for qubit in targets:
                    prob = results[ro_pulses[qubit].serial][0].probability(state=1)
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
