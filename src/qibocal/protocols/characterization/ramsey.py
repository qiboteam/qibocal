from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab.execution_parameters import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log

from .utils import GHZ_TO_HZ, HZ_TO_GHZ, V_TO_UV


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

    frequency: dict[QubitId, float] = field(metadata=dict(update="drive_frequency"))
    """Drive frequency [GHz] for each qubit."""
    t2: dict[QubitId, float]
    """T2 for each qubit [ns]."""
    delta_phys: dict[QubitId, float]
    """Drive frequency [Hz] correction for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


RamseyType = np.dtype(
    [("wait", np.float64), ("msr", np.float64), ("phase", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class RamseyData(Data):
    """Ramsey acquisition outputs."""

    n_osc: int
    """Number of oscillations for detuning."""
    t_max: int
    """Final delay between RX(pi/2) pulses in ns."""
    detuning_sign: int
    """Sign for induced detuning."""
    qubit_freqs: dict[QubitId, float] = field(default_factory=dict)
    """Qubit freqs for each qubit."""
    data: dict[QubitId, npt.NDArray[RamseyType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, wait, msr, phase):
        """Store output for single qubit."""
        # to be able to handle the non-sweeper case
        shape = (1,) if np.isscalar(wait) else wait.shape
        ar = np.empty(shape, dtype=RamseyType)
        ar["wait"] = wait
        ar["msr"] = msr
        ar["phase"] = phase
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


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

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include wait time and t_max
    data = RamseyData(
        n_osc=params.n_osc,
        t_max=params.delay_between_pulses_end,
        detuning_sign=+1,
        qubit_freqs=freqs,
    )

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
            for qubit in qubits:
                result = results[ro_pulses[qubit].serial]
                data.register_qubit(
                    qubit, wait=wait, msr=result.magnitude, phase=result.phase
                )

    else:
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
            data.register_qubit(
                qubit, wait=waits, msr=result.magnitude, phase=result.phase
            )
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
    qubits = data.qubits

    t2s = {}
    corrected_qubit_frequencies = {}
    freqs_detuning = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        voltages = qubit_data.msr * V_TO_UV
        times = qubit_data.wait
        qubit_freq = data.qubit_freqs[qubit]

        try:
            y_max = np.max(voltages)
            y_min = np.min(voltages)
            y = (voltages - y_min) / (y_max - y_min)
            x_max = np.max(times)
            x_min = np.min(times)
            x = (times - x_min) / (x_max - x_min)

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
                (delta_fitting - data.n_osc / data.t_max) * GHZ_TO_HZ
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
        corrected_qubit_frequencies[qubit] = corrected_qubit_frequency * HZ_TO_GHZ
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

    qubit_data = data[qubit]
    fig.add_trace(
        go.Scatter(
            x=qubit_data.wait,
            y=qubit_data.msr * V_TO_UV,
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        )
    )

    # add fitting trace
    waitrange = np.linspace(
        min(qubit_data.wait),
        max(qubit_data.wait),
        2 * len(qubit_data),
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
        + (f"{qubit} | Drive_frequency: {fit.frequency[qubit] * GHZ_TO_HZ} Hz<br>")
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
