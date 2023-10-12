from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Qubits, Results, Routine

from .ramsey import (
    PERR_EXCEPTION,
    POPT_EXCEPTION,
    RamseyData,
    RamseyParameters,
    _update,
    fitting,
    ramsey_fit,
)
from .utils import GHZ_TO_HZ, V_TO_UV, table_dict, table_html


@dataclass
class RamseyMSRParameters(RamseyParameters):
    """Ramsey runcard inputs."""


@dataclass
class RamseyMSRResults(Results):
    """Ramsey outputs."""

    frequency: dict[QubitId, tuple[float, Optional[float]]]
    """Drive frequency [GHz] for each qubit."""
    t2: dict[QubitId, tuple[float, Optional[float]]]
    """T2 for each qubit [ns]."""
    delta_phys: dict[QubitId, tuple[float, Optional[float]]]
    """Drive frequency [Hz] correction for each qubit."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""


RamseyMSRType = np.dtype([("wait", np.float64), ("msr", np.float64)])
"""Custom dtype for coherence routines."""


@dataclass
class RamseyMSRData(RamseyData):
    """Ramsey acquisition outputs."""

    def register_qubit(self, qubit, wait, msr):
        """Store output for single qubit."""
        # to be able to handle the non-sweeper case
        shape = (1,) if np.isscalar(msr) else msr.shape
        ar = np.empty(shape, dtype=RamseyMSRType)
        ar["wait"] = wait
        ar["msr"] = msr
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)

    @property
    def waits(self):
        """
        Return a list with the waiting times without repetitions.
        """
        qubit = next(iter(self.data))
        return np.unique(self.data[qubit].wait)


def _acquisition(
    params: RamseyMSRParameters,
    platform: Platform,
    qubits: Qubits,
) -> RamseyMSRData:
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
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        freqs[qubit] = qubits[qubit].drive_frequency
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

    data = RamseyMSRData(
        n_osc=params.n_osc,
        t_max=params.delay_between_pulses_end,
        detuning_sign=+1,
        qubit_freqs=freqs,
    )

    if params.n_osc == 0:
        sweeper = Sweeper(
            Parameter.start,
            waits,
            [RX90_pulses2[qubit] for qubit in qubits],
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
                msr=result.magnitude,
            )

    else:
        for wait in waits:
            for qubit in qubits:
                RX90_pulses2[qubit].start = RX90_pulses1[qubit].finish + wait
                ro_pulses[qubit].start = RX90_pulses2[qubit].finish
                if params.n_osc != 0:
                    RX90_pulses2[qubit].relative_phase = (
                        RX90_pulses2[qubit].start
                        * (-2 * np.pi)
                        * (params.n_osc)
                        / params.delay_between_pulses_end
                    )

            results = platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                    averaging_mode=(AveragingMode.CYCLIC),
                ),
            )

            for qubit in qubits:
                result = results[ro_pulses[qubit].serial]
                data.register_qubit(
                    qubit,
                    wait=wait,
                    msr=result.magnitude,
                )
    return data


def _fit(data: RamseyMSRData) -> RamseyMSRResults:
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
    for qubit in qubits:
        qubit_data = data[qubit]
        qubit_freq = data.qubit_freqs[qubit]
        msr = qubit_data["msr"]
        try:
            popt, perr = fitting(waits, msr)
        except:
            popt = POPT_EXCEPTION
            perr = PERR_EXCEPTION

        delta_fitting = popt[2] / (2 * np.pi)
        delta_phys = data.detuning_sign * int(
            (delta_fitting - data.n_osc / data.t_max) * GHZ_TO_HZ
        )
        corrected_qubit_frequency = int(qubit_freq - delta_phys)
        t2 = popt[4]
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

    return RamseyMSRResults(freq_measure, t2_measure, delta_phys_measure, popts)


def _plot(data: RamseyMSRData, qubit, fit: RamseyMSRResults = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    qubit_data = data.data[qubit]
    waits = data.waits
    msr = qubit_data["msr"]
    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=msr * V_TO_UV,
                opacity=1,
                name="Voltage",
                showlegend=True,
                legendgroup="Voltage",
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
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                    float(fit.fitted_parameters[qubit][4]),
                )
                * V_TO_UV,
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            )
        )
        fitting_report = table_html(
            table_dict(
                qubit,
                [
                    "Delta Frequency [Hz]",
                    "Drive Frequency [Hz]",
                    "T2* [ns]",
                ],
                [
                    np.round(fit.delta_phys[qubit][0], 3),
                    np.round(fit.frequency[qubit][0], 3),
                    np.round(fit.t2[qubit][0], 3),
                ],
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR [uV]",
    )

    figures.append(fig)

    return figures, fitting_report


ramsey_msr = Routine(_acquisition, _fit, _plot, _update)
"""Ramsey Routine object."""
