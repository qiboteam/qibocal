from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import Custom, PulseSequence, ReadoutPulse
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Results, Routine
from qibocal.config import log

from ..utils import GHZ_TO_HZ, table_dict, table_html
from .ramsey import RamseyData, RamseyParameters, _update
from .utils import fitting, ramsey_fit, ramsey_sequence


@dataclass
class RamseySignalParameters(RamseyParameters):
    """Ramsey runcard inputs."""


@dataclass
class RamseySignalResults(Results):
    """Ramsey outputs."""

    frequency: dict[QubitId, tuple[float, Optional[float]]]
    """Drive frequency [GHz] for each qubit."""
    t2: dict[QubitId, tuple[float, Optional[float]]]
    """T2 for each qubit [ns]."""
    delta_phys: dict[QubitId, tuple[float, Optional[float]]]
    """Drive frequency [Hz] correction for each qubit."""
    delta_fitting: dict[QubitId, tuple[float, Optional[float]]]
    """Raw drive frequency [Hz] correction for each qubit.
       including the detuning."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""


RamseySignalType = np.dtype([("wait", np.float64), ("signal", np.float64)])
"""Custom dtype for coherence routines."""


@dataclass
class RamseySignalData(RamseyData):
    """Ramsey acquisition outputs."""

    def register_qubit(self, qubit, wait, signal):
        """Store output for single qubit."""
        # to be able to handle the non-sweeper case
        shape = (1,) if np.isscalar(signal) else signal.shape
        ar = np.empty(shape, dtype=RamseySignalType)
        ar["wait"] = wait
        ar["signal"] = signal
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: RamseySignalParameters,
    platform: Platform,
    targets: list[QubitId],
) -> RamseySignalData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    ro_pulses = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
    freqs = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit,
            start=RX90_pulses1[qubit].finish,
        )
        unpadded_ro_pulse = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        padded_ro_pulse = ReadoutPulse(
            start=unpadded_ro_pulse.start - RX90_pulses2[qubit].duration,
            duration=unpadded_ro_pulse.duration + RX90_pulses2[qubit].duration,
            amplitude=unpadded_ro_pulse.amplitude,
            frequency=unpadded_ro_pulse.frequency,
            relative_phase=unpadded_ro_pulse.relative_phase,
            shape=Custom(
                envelope_i=np.concatenate(
                    (
                        np.zeros(RX90_pulses2[qubit].duration),
                        unpadded_ro_pulse.envelope_waveform_i.data
                        / unpadded_ro_pulse.amplitude,
                    )
                ),
                envelope_q=np.concatenate(
                    (
                        np.zeros(RX90_pulses2[qubit].duration),
                        unpadded_ro_pulse.envelope_waveform_q.data
                        / unpadded_ro_pulse.amplitude,
                    )
                ),
            ),
            channel=unpadded_ro_pulse.channel,
            qubit=unpadded_ro_pulse.qubit,
        )
        ro_pulses[qubit] = padded_ro_pulse
        freqs[qubit] = qubits[qubit].drive_frequency

        if params.n_osc != 0:
            RX90_pulses1[qubit].frequency += (
                params.n_osc / params.delay_between_pulses_end * 1e9
            )
            RX90_pulses2[qubit].frequency += (
                params.n_osc / params.delay_between_pulses_end * 1e9
            )

        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:

    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    data = RamseySignalData(
        n_osc=params.n_osc,
        t_max=params.delay_between_pulses_end,
        detuning_sign=+1,
        qubit_freqs=freqs,
    )

<<<<<<<< HEAD:src/qibocal/protocols/characterization/alvaro/ramsey_signal.py
    sweeper = Sweeper(
        Parameter.start,
        waits,
        [RX90_pulses2[qubit] for qubit in qubits]
        + [ro_pulses[qubit] for qubit in qubits],
        type=SweeperType.ABSOLUTE,
    )

    # execute the sweep
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )
    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]
        # The probability errors are the standard errors of the binomial distribution
        data.register_qubit(
            qubit,
            wait=waits,
            signal=result.magnitude,
        )
========
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
                qubit,
                wait=waits,
                signal=result.magnitude,
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
                    qubit,
                    wait=wait,
                    signal=result.magnitude,
                )

>>>>>>>> main:src/qibocal/protocols/characterization/ramsey/ramsey_signal.py
    return data


def _fit(data: RamseySignalData) -> RamseySignalResults:
    r"""
    Fitting routine for Ramsey experiment. The used model is
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
            freq_measure[qubit] = (
                corrected_qubit_frequency,
                perr[2] * GHZ_TO_HZ / (2 * np.pi),
            )
            t2_measure[qubit] = (t2, perr[4] * (t2**2))
            popts[qubit] = popt
            delta_phys_measure[qubit] = (
                delta_phys,
                perr[2] * GHZ_TO_HZ / (2 * np.pi),
            )
            delta_fitting_measure[qubit] = (
                delta_fitting * GHZ_TO_HZ,
                perr[2] * GHZ_TO_HZ / (2 * np.pi),
            )
        except Exception as e:
            log.warning(f"Ramsey fitting failed for qubit {qubit} due to {e}.")

<<<<<<<< HEAD:src/qibocal/protocols/characterization/alvaro/ramsey_signal.py
        delta_fitting = popt[2] / (2 * np.pi)
        delta_phys = data.detuning_sign * int(
            (delta_fitting - data.n_osc / data.t_max) * GHZ_TO_HZ
        )
        corrected_qubit_frequency = int(qubit_freq - delta_phys)
        t2 = 1 / popt[4]  # to check
        freq_measure[qubit] = (
            corrected_qubit_frequency,
            perr[2] * GHZ_TO_HZ / (2 * np.pi * data.t_max),
        )
        t2_measure[qubit] = (t2, perr[4])
        popts[qubit] = popt
        delta_phys_measure[qubit] = (
            delta_phys,
            popt[2] * GHZ_TO_HZ / (2 * np.pi * data.t_max),
        )

    return RamseySignalResults(freq_measure, t2_measure, delta_phys_measure, popts)
========
    return RamseySignalResults(
        frequency=freq_measure,
        t2=t2_measure,
        delta_phys=delta_phys_measure,
        delta_fitting=delta_fitting_measure,
        fitted_parameters=popts,
    )
>>>>>>>> main:src/qibocal/protocols/characterization/ramsey/ramsey_signal.py


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
<<<<<<<< HEAD:src/qibocal/protocols/characterization/alvaro/ramsey_signal.py
                    np.round(fit.delta_phys[qubit][0], 3),
                    np.round(fit.frequency[qubit][0], 3),
                    fit.t2[qubit][0],
                    # np.round(fit.t2[qubit][0], 3),
========
                    np.round(fit.delta_phys[target][0], 3),
                    np.round(fit.delta_fitting[target][0], 3),
                    np.round(fit.frequency[target][0], 3),
                    np.round(fit.t2[target][0], 3),
>>>>>>>> main:src/qibocal/protocols/characterization/ramsey/ramsey_signal.py
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


ramsey_signal = Routine(_acquisition, _fit, _plot, _update)
"""Ramsey Routine object."""
