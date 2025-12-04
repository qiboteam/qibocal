from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Readout, Sweeper

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import magnitude

from ... import update
from ..utils import readout_frequency, table_dict, table_html
from .utils import fitting, process_fit, ramsey_fit, ramsey_sequence

__all__ = [
    "ramsey_signal",
    "RamseySignalParameters",
    "RamseySignalData",
    "_update",
    "RamseySignalResults",
]


@dataclass
class RamseySignalParameters(Parameters):
    """Ramsey runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""
    detuning: Optional[int] = None
    """Frequency detuning [Hz] (optional).
        If 0 standard Ramsey experiment is performed."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""


@dataclass
class RamseySignalResults(Results):
    """Ramsey outputs."""

    detuning: float
    """Qubit frequency detuning."""
    frequency: dict[QubitId, Union[float, list[float]]]
    """Drive frequency [GHz] for each qubit."""
    t2: dict[QubitId, Union[float, list[float]]]
    """T2 for each qubit [ns]."""
    delta_phys: dict[QubitId, Union[float, list[float]]]
    """Drive frequency [Hz] correction for each qubit."""
    delta_fitting: dict[QubitId, Union[float, list[float]]]
    """Raw drive frequency [Hz] correction for each qubit.
       including the detuning."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""


RamseySignalType = np.dtype([("wait", np.float64), ("signal", np.float64)])
"""Custom dtype for coherence routines."""


@dataclass
class RamseySignalData(Data):
    """Ramsey acquisition outputs."""

    detuning: Optional[int] = None
    """Frequency detuning [Hz]."""
    qubit_freqs: dict[QubitId, float] = field(default_factory=dict)
    """Qubit freqs for each qubit."""
    data: dict[QubitId, npt.NDArray[RamseySignalType]] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def waits(self):
        """
        Return a list with the waiting times without repetitions.
        """
        qubit = next(iter(self.data))
        return np.unique(self.data[qubit].wait)


def _acquisition(
    params: RamseySignalParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RamseySignalData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    # define the parameter to sweep and its range:

    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    data = RamseySignalData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.config(platform.qubits[qubit].drive).frequency
            for qubit in targets
        },
    )

    updates = []
    updates += [
        {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
        for q in targets
    ]
    if params.detuning is not None:
        for qubit in targets:
            channel = platform.qubits[qubit].drive
            f0 = platform.config(channel).frequency
            updates.append({channel: {"frequency": f0 + params.detuning}})

    if not params.unrolling:
        sequence, delays = ramsey_sequence(platform, targets)
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
            updates=updates,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
        for qubit in targets:
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            result = results[ro_pulse.id]
            # The probability errors are the standard errors of the binomial distribution
            data.register_qubit(
                RamseySignalType,
                (qubit),
                dict(
                    wait=waits,
                    signal=magnitude(result),
                ),
            )

    else:
        sequences, all_ro_pulses = [], []
        for wait in waits:
            sequence, _ = ramsey_sequence(platform, targets, wait)
            sequences.append(sequence)
            all_ro_pulses.append(
                {
                    qubit: [
                        pulse
                        for pulse in list(
                            sequence.channel(platform.qubits[qubit].acquisition)
                        )
                        if isinstance(pulse, Readout)
                    ][0]
                    for qubit in targets
                }
            )

        results = platform.execute(
            sequences,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=updates,
        )

        for wait, ro_pulses in zip(waits, all_ro_pulses):
            for qubit in targets:
                result = results[ro_pulses[qubit].id]
                data.register_qubit(
                    RamseySignalType,
                    (qubit),
                    dict(
                        wait=np.array([wait]),
                        signal=np.array([magnitude(result)]),
                    ),
                )

    return data


def _fit(data: RamseySignalData) -> RamseySignalResults:
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
        signal = qubit_data["signal"]
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

    return RamseySignalResults(
        detuning=data.detuning,
        frequency=freq_measure,
        t2=t2_measure,
        delta_phys=delta_phys_measure,
        delta_fitting=delta_fitting_measure,
        fitted_parameters=popts,
    )


def _plot(data: RamseySignalData, target: QubitId, fit: RamseySignalResults = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    qubit_data = data.data[target]
    waits = data.waits
    signal = qubit_data["signal"]
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
                ],
                [
                    np.round(fit.delta_phys[target][0], 3),
                    np.round(fit.delta_fitting[target][0], 3),
                    np.round(fit.frequency[target][0], 3),
                    np.round(fit.t2[target][0], 3),
                ],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Signal [a.u.]",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: RamseySignalResults, platform: CalibrationPlatform, target: QubitId
):
    if results.detuning is not None:
        update.drive_frequency(results.frequency[target][0], platform, target)
        platform.calibration.single_qubits[
            target
        ].qubit.frequency_01 = results.frequency[target][0]
    else:
        update.t2(results.t2[target], platform, target)


ramsey_signal = Routine(_acquisition, _fit, _plot, _update)
"""Ramsey Routine object."""
