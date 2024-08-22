from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Routine
from qibocal.config import log

from ..utils import GHZ_TO_HZ, chi2_reduced, table_dict, table_html
from .ramsey import (
    RamseySignalData,
    RamseySignalParameters,
    RamseySignalResults,
    _update,
)
from .utils import fitting, ramsey_fit, ramsey_sequence

COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"


@dataclass
class RamseyZZParameters(RamseySignalParameters):
    """Ramsey runcard inputs."""

    target_qubit: Optional[QubitId] = None


@dataclass
class RamseyZZResults(RamseySignalResults):
    """Ramsey outputs."""

    chi2: dict[QubitId, tuple[float, Optional[float]]]
    """Chi squared estimate mean value and error. """

    def __contains__(self, key: QubitId):
        return True


RamseyZZType = np.dtype(
    [("wait", np.float64), ("prob", np.float64), ("errors", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class RamseyZZData(RamseySignalData):
    """Ramsey acquisition outputs."""

    data: dict[tuple[QubitId, str], npt.NDArray[RamseyZZType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _acquisition(
    params: RamseyZZParameters,
    platform: Platform,
    targets: list[QubitId],
) -> RamseyZZData:
    """Data acquisition for Ramsey Experiment (detuned).

    The protocol consists in applying the following pulse sequence
    RX90 - wait - RX90 - MZ
    for different waiting times `wait`.
    The range of waiting times is defined through the attributes
    `delay_between_pulses_*` available in `RamseyParameters`. The final range
    will be constructed using `np.arange`.
    It is possible to detune the drive frequency using the parameter `detuning` in
    RamseyParameters which will increment the drive frequency accordingly.
    Currently when `detuning==0` it will be performed a sweep over the waiting values
    if `detuning` is not zero, all sequences with different waiting value will be
    executed sequentially. By providing the option `unrolling=True` in RamseyParameters
    the sequences will be unrolled when the frequency is detuned.
    The following protocol will display on the y-axis the probability of finding the ground
    state, therefore it is advise to execute it only after having performed the single
    shot classification. Error bars are provided as binomial distribution error.
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

    sequence_I = PulseSequence()
    sequence_X = PulseSequence()

    data = RamseyZZData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.qubits[qubit].native_gates.RX.frequency for qubit in targets
        },
    )

    if not params.unrolling:
        for qubit in targets:
            sequence_I += ramsey_sequence(
                platform=platform, qubit=qubit, detuning=params.detuning
            )
            sequence_X += ramsey_sequence(
                platform=platform,
                qubit=qubit,
                detuning=params.detuning,
                target_qubit=params.target_qubit,
            )

        sweeper_I = Sweeper(
            Parameter.start,
            waits,
            [sequence_I.get_qubit_pulses(qubit).qd_pulses[-1] for qubit in targets],
            type=SweeperType.ABSOLUTE,
        )

        sweeper_X = Sweeper(
            Parameter.start,
            waits,
            [sequence_X.get_qubit_pulses(qubit).qd_pulses[-1] for qubit in targets],
            type=SweeperType.ABSOLUTE,
        )

        # execute the sweep
        results_I = platform.sweep(
            sequence_I,
            options,
            sweeper_I,
        )
        results_X = platform.sweep(
            sequence_X,
            options,
            sweeper_X,
        )
        for qubit in targets:
            probs_I = results_I[qubit].probability(state=1)
            # The probability errors are the standard errors of the binomial distribution
            errors_I = [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs_I]
            probs_X = results_X[qubit].probability(state=1)
            # The probability errors are the standard errors of the binomial distribution
            errors_X = [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs_X]
            data.register_qubit(
                RamseyZZType,
                (qubit, "I"),
                dict(
                    wait=waits,
                    prob=probs_I,
                    errors=errors_I,
                ),
            )

            data.register_qubit(
                RamseyZZType,
                (qubit, "X"),
                dict(
                    wait=waits,
                    prob=probs_X,
                    errors=errors_X,
                ),
            )

    if params.unrolling:
        sequences_I, all_ro_pulses_I = [], []
        sequences_X, all_ro_pulses_X = [], []
        for wait in waits:
            sequence_I = PulseSequence()
            sequence_X = PulseSequence()
            for qubit in targets:
                sequence_I += ramsey_sequence(
                    platform=platform, qubit=qubit, wait=wait, detuning=params.detuning
                )
                sequence_X += ramsey_sequence(
                    platform=platform,
                    qubit=qubit,
                    wait=wait,
                    detuning=params.detuning,
                    target_qubit=params.target_qubit,
                )

            sequences_I.append(sequence_I)
            all_ro_pulses_I.append(sequence_I.ro_pulses)
            sequences_X.append(sequence_X)
            all_ro_pulses_X.append(sequence_X.ro_pulses)

        results_I = platform.execute_pulse_sequences(sequences_I, options)
        results_X = platform.execute_pulse_sequences(sequences_X, options)

        # We dont need ig as every serial is different
        for ig, (wait, ro_pulses) in enumerate(zip(waits, all_ro_pulses_I)):
            for qubit in targets:
                serial = ro_pulses[qubit].serial
                if params.unrolling:
                    result = results_I[serial][0]
                else:
                    result = results_I[ig][serial]
                prob = result.probability()
                error = np.sqrt(prob * (1 - prob) / params.nshots)
                data.register_qubit(
                    RamseyZZType,
                    (qubit, "I"),
                    dict(
                        wait=np.array([wait]),
                        prob=np.array([prob]),
                        errors=np.array([error]),
                    ),
                )

        for ig, (wait, ro_pulses) in enumerate(zip(waits, all_ro_pulses_X)):
            for qubit in targets:
                serial = ro_pulses[qubit].serial
                if params.unrolling:
                    result = results_X[serial][0]
                else:
                    result = results_X[ig][serial]
                prob = result.probability()
                error = np.sqrt(prob * (1 - prob) / params.nshots)
                data.register_qubit(
                    RamseyZZType,
                    (qubit, "X"),
                    dict(
                        wait=np.array([wait]),
                        prob=np.array([prob]),
                        errors=np.array([error]),
                    ),
                )

    return data


def _fit(data: RamseyZZData) -> RamseyZZResults:
    r"""Fitting routine for Ramsey experiment. The used model is
    .. math::

        y = p_0 + p_1 sin \Big(p_2 x + p_3 \Big) e^{-x p_4}.
    """
    qubits = data.qubits
    waits = data.waits
    popts = {}
    freq_measure = {}
    t2_measure = {}
    delta_phys_measure = {}
    delta_fitting_measure = {}
    chi2 = {}
    for qubit in qubits:
        for setup in ["I", "X"]:
            qubit_data = data[qubit, setup]
            qubit_freq = data.qubit_freqs[qubit]
            probs = qubit_data["prob"]
            try:
                popt, perr = fitting(waits, probs, qubit_data.errors)

                delta_fitting = popt[2] / (2 * np.pi)
                sign = np.sign(data.detuning) if data.detuning != 0 else 1
                delta_phys = int(
                    sign * (delta_fitting * GHZ_TO_HZ - np.abs(data.detuning))
                )
                corrected_qubit_frequency = int(qubit_freq - delta_phys)
                t2 = 1 / popt[4]
                # TODO: check error formula
                freq_measure[qubit, setup] = (
                    corrected_qubit_frequency,
                    perr[2] * GHZ_TO_HZ / (2 * np.pi),
                )
                t2_measure[qubit, setup] = (t2, perr[4] * (t2**2))
                popts[qubit, setup] = popt
                # TODO: check error formula
                delta_phys_measure[qubit, setup] = (
                    -delta_phys,
                    perr[2] * GHZ_TO_HZ / (2 * np.pi),
                )
                delta_fitting_measure[qubit, setup] = (
                    -delta_fitting * GHZ_TO_HZ,
                    perr[2] * GHZ_TO_HZ / (2 * np.pi),
                )
                chi2[qubit, setup] = (
                    chi2_reduced(
                        probs,
                        ramsey_fit(waits, *popts[qubit]),
                        qubit_data.errors,
                    ),
                    np.sqrt(2 / len(probs)),
                )
            except Exception as e:
                log.warning(f"Ramsey fitting failed for qubit {qubit} due to {e}.")
    return RamseyZZResults(
        frequency=freq_measure,
        t2=t2_measure,
        delta_phys=delta_phys_measure,
        delta_fitting=delta_fitting_measure,
        fitted_parameters=popts,
        chi2=chi2,
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
                    "ZZ  [kHz]",
                ],
                [
                    np.round(
                        fit.delta_fitting[target, "X"][0]
                        - fit.delta_fitting[target, "I"][0]
                    )
                    * 1e-3,
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
