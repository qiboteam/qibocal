from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log

from ..utils import GHZ_TO_HZ, table_dict, table_html
from .utils import fitting, ramsey_fit, ramsey_sequence


@dataclass
class RamseySignalParameters(Parameters):
    """Ramsey runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""
    detuning: Optional[int] = 0
    """Frequency detuning [Hz] (optional).
        If 0 standard Ramsey experiment is performed."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""


@dataclass
class RamseySignalResults(Results):
    """Ramsey outputs."""

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

    detuning: int
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
    platform: Platform,
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

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = RamseySignalData(
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
            [
                sequence.get_qubit_pulses(qubit).qd_pulses[-1] for qubit in targets
            ],  # TODO: check if it is correct
            type=SweeperType.ABSOLUTE,
        )

        # execute the sweep
        results = platform.sweep(
            sequence,
            options,
            sweeper,
        )
        for qubit in targets:
            result = results[sequence.get_qubit_pulses(qubit).ro_pulses[0].serial]
            # The probability errors are the standard errors of the binomial distribution
            data.register_qubit(
                RamseySignalType,
                (qubit),
                dict(
                    wait=waits,
                    signal=result.magnitude,
                ),
            )

    else:
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

        # We dont need ig as everty serial is different
        for ig, (wait, ro_pulses) in enumerate(zip(waits, all_ro_pulses)):
            for qubit in targets:
                serial = ro_pulses[qubit].serial
                if params.unrolling:
                    result = results[serial][0]
                else:
                    result = results[ig][serial]
                data.register_qubit(
                    RamseySignalType,
                    (qubit),
                    dict(
                        wait=np.array([wait]),
                        signal=np.array([result.magnitude]),
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
            delta_fitting = popt[2] / (2 * np.pi)
            sign = np.sign(data.detuning) if data.detuning != 0 else 1
            delta_phys = int(sign * (delta_fitting * GHZ_TO_HZ - np.abs(data.detuning)))
            corrected_qubit_frequency = int(qubit_freq - delta_phys)
            t2 = 1 / popt[4]
            freq_measure[qubit] = [
                corrected_qubit_frequency,
                perr[2] * GHZ_TO_HZ / (2 * np.pi),
            ]
            t2_measure[qubit] = [t2, perr[4] * (t2**2)]
            popts[qubit] = popt
            delta_phys_measure[qubit] = [
                -delta_phys,
                perr[2] * GHZ_TO_HZ / (2 * np.pi),
            ]
            delta_fitting_measure[qubit] = [
                -delta_fitting * GHZ_TO_HZ,
                perr[2] * GHZ_TO_HZ / (2 * np.pi),
            ]
        except Exception as e:
            log.warning(f"Ramsey fitting failed for qubit {qubit} due to {e}.")

    return RamseySignalResults(
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


def _update(results: RamseySignalResults, platform: Platform, target: QubitId):
    update.drive_frequency(results.frequency[target][0], platform, target)


ramsey_signal = Routine(_acquisition, _fit, _plot, _update)
"""Ramsey Routine object."""
