from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html
from qibocal.result import unpack

from .ramsey_acquisition import (
    RamseyData,
    RamseyParameters,
    RamseyResults,
    execute_experiment,
    ramsey_sequence,
)
from .utils import fitting, process_fit, ramsey_fit, ramsey_update

__all__ = ["ramsey_signal"]


RamseySignalType = np.dtype(
    [("wait", np.float64), ("i", np.float64), ("q", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class RamseySignalData(RamseyData):
    """Ramsey acquisition outputs."""

    data: dict[QubitId, npt.NDArray[RamseySignalType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RamseyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RamseySignalData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    # define the parameter to sweep and its range:

    data = RamseySignalData(
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
        return_probs=False,
    )

    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        i, q = unpack(results[ro_pulse.id])
        data.register_qubit(
            RamseySignalType,
            (qubit),
            dict(
                wait=np.arange(*params.delay_range),
                i=i,
                q=q,
            ),
        )

    return data


def _fit(data: RamseySignalData) -> RamseyResults:
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
        qubit_freq = data.qubit_freqs[qubit]
        signal = data.compute_qubit_signal(qubit)
        try:
            popt, perr = fitting(waits, signal)
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


def _plot(data: RamseySignalData, target: QubitId, fit: Optional[RamseyResults] = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    waits = data.waits
    signal = data.compute_qubit_signal(target)
    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=signal,
                opacity=1,
                name="Signal",
                showlegend=True,
                legendgroup="Signal",
                mode="lines",
            ),
        ]
    )

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=waits,
                y=ramsey_fit(
                    waits,
                    *fit.fitted_parameters[target],
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
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


ramsey_signal = Routine(_acquisition, _fit, _plot, ramsey_update)
"""Ramsey Routine object."""
