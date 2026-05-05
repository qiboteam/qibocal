from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ..utils import table_dict, table_html
from .ramsey_acquisition import (
    RamseyData,
    RamseyParameters,
    RamseyResults,
    execute_experiment,
    ramsey_sequence,
)
from .utils import fitting, process_fit, ramsey_fit, ramsey_update

__all__ = ["ramsey"]


RamseyProbType = np.dtype([("wait", np.float64), ("prob", np.float64)])
"""Custom dtype for coherence routines."""


@dataclass
class RamseyProbData(RamseyData):
    """Ramsey acquisition outputs."""

    data: dict[QubitId, npt.NDArray[RamseyProbType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RamseyParameters,
    platform: CalibrationPlatform,
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

    data = RamseyProbData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.config(platform.qubits[qubit].drive).frequency
            for qubit in targets
        },
    )

    sequence, delays = ramsey_sequence(platform, targets)

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
            RamseyProbType,
            (qubit),
            dict(
                wait=np.arange(*params.delay_range),
                prob=results[ro_pulse.id],
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
    for qubit in qubits:
        qubit_data = data[qubit]
        qubit_freq = data.qubit_freqs[qubit]
        probs = qubit_data["prob"]
        try:
            popt, perr = fitting(waits, probs)
            (
                freq_measure[qubit],
                t2_measure[qubit],
                delta_phys_measure[qubit],
                delta_fitting_measure[qubit],
                popts[qubit],
            ) = process_fit(popt, perr, qubit_freq, data.detuning)

        except Exception as e:
            log.warning(f"Ramsey fitting failed for qubit {qubit} due to {e}.")
    return RamseyResults(
        detuning=data.detuning,
        frequency=freq_measure,
        t2=t2_measure,
        delta_phys=delta_phys_measure,
        delta_fitting=delta_fitting_measure,
        fitted_parameters=popts,
    )


def _plot(data: RamseyData, target: QubitId, fit: RamseyResults = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    qubit_data = data.data[target]
    waits = data.waits
    probs = qubit_data["prob"]
    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=probs,
                opacity=1,
                name="Probability of State 1",
                showlegend=True,
                legendgroup="Probability of State 1",
                mode="lines",
            ),
        ]
    )

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=waits,
                y=ramsey_fit(waits, *fit.fitted_parameters[target]),
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
                ],
                [
                    fit.delta_phys[target],
                    fit.delta_fitting[target],
                    fit.frequency[target],
                    fit.t2[target],
                ],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Excited state probability",
    )

    figures.append(fig)

    return figures, fitting_report


ramsey = Routine(_acquisition, _fit, _plot, ramsey_update)
"""Ramsey Routine object."""
