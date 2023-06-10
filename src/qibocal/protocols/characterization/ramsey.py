from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits


@dataclass
class RamseyParameters(Parameters):
    """Ramsey runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""
    n_osc: Optional[int] = 0
    """Number of oscillations to induce detuning (optional).
        If 0 standard Ramsey experiment is performed."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class RamseyResults(Results):
    """Ramsey outputs."""

    frequency: Dict[QubitId, float] = field(metadata=dict(update="drive_frequency"))
    """Drive frequency [GHz] for each qubit."""
    t2: Dict[QubitId, float]
    """T2 for each qubit [ns]."""
    delta_phys: Dict[QubitId, float]
    """Drive frequency [Hz] correction for each qubit."""
    fitted_parameters: Dict[QubitId, Dict[str, float]]
    """Raw fitting output."""


class RamseyData(DataUnits):
    """Ramsey acquisition outputs."""

    def __init__(self, n_osc, t_max, detuning_sign):
        super().__init__(
            name="data",
            quantities={"wait": "ns", "qubit_freqs": "Hz"},
            options=[
                "qubit",
            ],
        )

        self._n_osc = n_osc
        self._t_max = t_max
        self._detuning_sign = detuning_sign

    @property
    def n_osc(self):
        """Number of oscillations for detuning."""
        return self._n_osc

    @property
    def t_max(self):
        """Final delay between RX(pi/2) pulses in ns."""
        return self._t_max

    @property
    def detuning_sign(self):
        """Sign for induced detuning."""
        return self._detuning_sign


def _acquisition(
    params: RamseyParameters,
    platform: Platform,
    qubits: Qubits,
) -> RamseyData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    ro_pulses = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
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

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include wait time and t_max
    data = RamseyData(params.n_osc, params.delay_between_pulses_end, detuning_sign=+1)

    if params.n_osc != 0:
        # sweep the parameter
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

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
            )
            for qubit, ro_pulse in ro_pulses.items():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].serialize
                r.update(
                    {
                        "wait[ns]": wait,
                        "qubit_freqs[Hz]": qubits[qubit].drive_frequency,
                        "qubit": qubit,
                    }
                )
                data.add_data_from_dict(r)

    else:
        sweeper = Sweeper(
            Parameter.start,
            waits,
            [RX90_pulses2[qubit] for qubit in qubits],
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
            # average msr, phase, i and q over the number of shots defined in the runcard
            r = results[ro_pulses[qubit].serial].serialize
            r.update(
                {
                    "wait[ns]": waits,
                    "qubit_freqs[Hz]": len(waits) * [qubits[qubit].drive_frequency],
                    "qubit": len(waits) * [qubit],
                }
            )
            data.add_data_from_dict(r)

    return data


def ramsey_fit(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   DeltaFreq                    : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(x * p2 + p3) * np.exp(-x * p4)


def _fit(data: RamseyData) -> RamseyResults:
    r"""
    Fitting routine for Ramsey experiment. The used model is
    .. math::
        y = p_0 + p_1 sin \Big(p_2 x + p_3 \Big) e^{-x p_4}.
    """
    qubits = data.df["qubit"].unique()

    t2s = {}
    corrected_qubit_frequencies = {}
    freqs_detuning = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data_df = data.df[data.df["qubit"] == qubit]
        voltages = qubit_data_df["MSR"].pint.to("uV").pint.magnitude
        times = qubit_data_df["wait"].pint.to("ns").pint.magnitude
        qubit_freq = qubit_data_df["qubit_freqs"].pint.to("Hz").pint.magnitude.unique()

        try:
            y_max = np.max(voltages.values)
            y_min = np.min(voltages.values)
            y = (voltages.values - y_min) / (y_max - y_min)
            x_max = np.max(times.values)
            x_min = np.min(times.values)
            x = (times.values - x_min) / (x_max - x_min)

            ft = np.fft.rfft(y)
            freqs = np.fft.rfftfreq(len(y), x[1] - x[0])
            mags = abs(ft)
            index = np.argmax(mags) if np.argmax(mags) != 0 else np.argmax(mags[1:]) + 1
            f = freqs[index] * 2 * np.pi
            p0 = [
                0.5,
                0.5,
                f,
                0,
                0,
            ]
            popt = curve_fit(
                ramsey_fit,
                x,
                y,
                p0=p0,
                maxfev=2000000,
                bounds=(
                    [0, 0, 0, -np.pi, 0],
                    [1, 1, np.inf, np.pi, np.inf],
                ),
            )[0]
            popt = [
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
                popt[2] / (x_max - x_min),
                popt[3] - x_min * popt[2] / (x_max - x_min),
                popt[4] / (x_max - x_min),
            ]
            delta_fitting = popt[2] / (2 * np.pi)
            delta_phys = data.detuning_sign * int(
                (delta_fitting - data.n_osc / data.t_max) * 1e9
            )
            # FIXME: for qblox the correct formula is the following (there is a bug related to the phase)
            # corrected_qubit_frequency = int(qubit_freq + delta_phys)
            corrected_qubit_frequency = int(qubit_freq - delta_phys)
            t2 = 1.0 / popt[4]

        except Exception as e:
            log.warning(f"ramsey_fit: the fitting was not succesful. {e}")
            popt = [0] * 5
            t2 = 5.0
            print(qubit_freq)
            corrected_qubit_frequency = int(qubit_freq)
            delta_phys = 0

        fitted_parameters[qubit] = popt
        corrected_qubit_frequencies[qubit] = corrected_qubit_frequency / 1e9
        t2s[qubit] = t2
        freqs_detuning[qubit] = delta_phys

    return RamseyResults(
        corrected_qubit_frequencies, t2s, freqs_detuning, fitted_parameters
    )


def _plot(data: RamseyData, fit: RamseyResults, qubit):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    qubit_data = data.df[data.df["qubit"] == qubit]

    fig.add_trace(
        go.Scatter(
            x=qubit_data["wait"].pint.magnitude,
            y=qubit_data["MSR"].pint.to("uV").pint.magnitude,
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        )
    )

    # add fitting trace
    if len(data) > 0:
        waitrange = np.linspace(
            min(qubit_data["wait"].pint.to("ns").pint.magnitude),
            max(qubit_data["wait"].pint.to("ns").pint.magnitude),
            2 * len(data),
        )

        fig.add_trace(
            go.Scatter(
                x=waitrange,
                y=ramsey_fit(
                    waitrange,
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                    float(fit.fitted_parameters[qubit][4]),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            )
        )
        fitting_report = (
            fitting_report
            + (f"{qubit} | Delta_frequency: {fit.delta_phys[qubit]:,.1f} Hz<br>")
            + (f"{qubit} | Drive_frequency: {fit.frequency[qubit] * 1e9} Hz<br>")
            + (f"{qubit} | T2: {fit.t2[qubit]:,.0f} ns.<br><br>")
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


ramsey = Routine(_acquisition, _fit, _plot)
"""Ramsey Routine object."""
