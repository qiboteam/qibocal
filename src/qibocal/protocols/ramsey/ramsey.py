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

from ..utils import chi2_reduced, table_dict, table_html
from .ramsey_signal import (
    RamseySignalData,
    RamseySignalParameters,
    RamseySignalResults,
    _update,
)
from .utils import fitting, process_fit, ramsey_fit, ramsey_sequence

COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"


@dataclass
class RamseyParameters(RamseySignalParameters):
    """Ramsey runcard inputs."""


@dataclass
class RamseyResults(RamseySignalResults):
    """Ramsey outputs."""

    chi2: dict[QubitId, tuple[float, Optional[float]]]
    """Chi squared estimate mean value and error. """


RamseyType = np.dtype(
    [("wait", np.float64), ("prob", np.float64), ("errors", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class RamseyData(RamseySignalData):
    """Ramsey acquisition outputs."""

    data: dict[QubitId, npt.NDArray[RamseyType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RamseyParameters,
    platform: Platform,
    targets: list[QubitId],
) -> RamseyData:
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
        # wait time between RX90 pulses
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

    sequence = PulseSequence()

    data = RamseyData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.qubits[qubit].native_gates.RX.frequency for qubit in targets
        },
    )

    if not params.unrolling:
        sequence = PulseSequence()
        for qubit in targets:
            sequence += ramsey_sequence(
                platform=platform, qubit=qubit, detuning=params.detuning
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
            # The probability errors are the standard errors of the binomial distribution
            errors = [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs]
            data.register_qubit(
                RamseyType,
                (qubit),
                dict(
                    wait=waits,
                    prob=probs,
                    errors=errors,
                ),
            )

    if params.unrolling:
        sequences, all_ro_pulses = [], []
        for wait in waits:
            sequence = PulseSequence()
            for qubit in targets:
                sequence += ramsey_sequence(
                    platform=platform, qubit=qubit, wait=wait, detuning=params.detuning
                )

            sequences.append(sequence)
            all_ro_pulses.append(sequence.ro_pulses)

        results = platform.execute_pulse_sequences(sequences, options)

        # We dont need ig as every serial is different
        for ig, (wait, ro_pulses) in enumerate(zip(waits, all_ro_pulses)):
            for qubit in targets:
                serial = ro_pulses[qubit].serial
                if params.unrolling:
                    result = results[serial][0]
                else:
                    result = results[ig][serial]
                prob = result.probability()
                error = np.sqrt(prob * (1 - prob) / params.nshots)
                data.register_qubit(
                    RamseyType,
                    (qubit),
                    dict(
                        wait=np.array([wait]),
                        prob=np.array([prob]),
                        errors=np.array([error]),
                    ),
                )

    return data


def _fit(data: RamseyData) -> RamseyResults:
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
        qubit_data = data[qubit]
        qubit_freq = data.qubit_freqs[qubit]
        probs = qubit_data["prob"]
        try:
            popt, perr = fitting(waits, probs, qubit_data.errors)
            (
                freq_measure[qubit],
                t2_measure[qubit],
                delta_phys_measure[qubit],
                delta_fitting_measure[qubit],
                popts[qubit],
            ) = process_fit(popt, perr, qubit_freq, data.detuning)

            chi2[qubit] = (
                chi2_reduced(
                    probs,
                    ramsey_fit(waits, *popts[qubit]),
                    qubit_data.errors,
                ),
                np.sqrt(2 / len(probs)),
            )
        except Exception as e:
            log.warning(f"Ramsey fitting failed for qubit {qubit} due to {e}.")
    return RamseyResults(
        detuning=data.detuning,
        frequency=freq_measure,
        t2=t2_measure,
        delta_phys=delta_phys_measure,
        delta_fitting=delta_fitting_measure,
        fitted_parameters=popts,
        chi2=chi2,
    )


def _plot(data: RamseyData, target: QubitId, fit: RamseyResults = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    qubit_data = data.data[target]
    waits = data.waits
    probs = qubit_data["prob"]
    error_bars = qubit_data["errors"]
    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=probs,
                opacity=1,
                name="Probability of State 0",
                showlegend=True,
                legendgroup="Probability of State 0",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((waits, waits[::-1])),
                y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=waits,
                y=ramsey_fit(
                    waits,
                    float(fit.fitted_parameters[target][0]),
                    float(fit.fitted_parameters[target][1]),
                    float(fit.fitted_parameters[target][2]),
                    float(fit.fitted_parameters[target][3]),
                    float(fit.fitted_parameters[target][4]),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            )
        )
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Delta Frequency [Hz]",
                    "Delta Frequency (with detuning) [Hz]",
                    "Drive Frequency [Hz]",
                    "T2* [ns]",
                    "chi2 reduced",
                ],
                [
                    fit.delta_phys[target],
                    fit.delta_fitting[target],
                    fit.frequency[target],
                    fit.t2[target],
                    fit.chi2[target],
                ],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Ground state probability",
    )

    figures.append(fig)

    return figures, fitting_report


ramsey = Routine(_acquisition, _fit, _plot, _update)
"""Ramsey Routine object."""
