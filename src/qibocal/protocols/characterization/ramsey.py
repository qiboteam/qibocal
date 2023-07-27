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
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.bootstrap import bootstrap
from qibocal.config import log

from .utils import GHZ_TO_HZ, V_TO_UV


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
    nboot: int = 0
    """Number of bootstrap samples"""


@dataclass
class RamseyResults(Results):
    """Ramsey outputs."""

    frequency: dict[QubitId, float] = field(metadata=dict(update="drive_frequency"))
    """Drive frequency [GHz] for each qubit."""
    error_frequency: dict[QubitId, float]
    """Error of the drive frequency [GHz]."""
    t2: dict[QubitId, float]
    """T2 for each qubit [ns]."""
    error_t2: dict[QubitId, float]
    """Error T2 [ns]."""
    delta_phys: dict[QubitId, float]
    """Drive frequency [Hz] correction for each qubit."""
    error_delta_phys: dict[QubitId, float]
    """Error of the drive frequency [Hz]."""
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
    nboot: int
    """Number of bootstrap samples"""
    qubit_freqs: dict[QubitId, float] = field(default_factory=dict)
    """Qubit freqs for each qubit."""
    data: dict[QubitId, npt.NDArray[RamseyType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, wait, msr, phase):
        """Store output for single qubit."""
        # to be able to handle the non-sweeper case
        ar = np.empty(msr.shape, dtype=RamseyType)
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

    data = RamseyData(
        n_osc=params.n_osc,
        t_max=params.delay_between_pulses_end,
        detuning_sign=+1,
        nboot=params.nboot,
        qubit_freqs=freqs,
    )

    if params.n_osc == 0 and params.nboot == 0:
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

            # execute the pulse sequence
            averaging_mode = AveragingMode.SINGLESHOT if params.nboot != 0 else AveragingMode.CYCLIC

            results = platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                    averaging_mode=averaging_mode,
                ),
            )
            for qubit in qubits:
                result = results[ro_pulses[qubit].serial]
                data.register_qubit(
                    qubit,
                    wait=wait,
                    msr=np.array([result.magnitude]),
                    phase=np.array([result.phase]),
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
    waits = np.unique(data[qubits[0]].wait)
    popts = {}
    freq_av = {}
    freq_err = {}
    t2_av = {}
    t2_err = {}
    delta_phys_av = {}
    delta_phys_err = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        qubit_freq = data.qubit_freqs[qubit]
        msrs = np.array(qubit_data[["msr"]].tolist()) * V_TO_UV

        msrs = msrs.reshape((len(waits), -1))
        t2s = []
        deltas_phys_list = []
        new_freqs = []
        if data.nboot != 0:
            bootstrap_samples = bootstrap(msrs, data.nboot)
            voltages = np.mean(bootstrap_samples, axis=1)
            for i in range(data.nboot):
                y = voltages[:, i]
                x = waits
                try:
                    popt = fitting(x, y)
                    delta_fitting = popt[2] / (2 * np.pi)
                    delta_phys = data.detuning_sign * int(
                        (delta_fitting - data.n_osc / data.t_max) * GHZ_TO_HZ
                    )

                    # FIXME: for qblox the correct formula is the following (there is a bug related to the phase)
                    # corrected_qubit_frequency = int(qubit_freq + delta_phys)
                    corrected_qubit_frequency = int(qubit_freq - delta_phys)
                    deltas_phys_list.append(delta_phys)
                    new_freqs.append(corrected_qubit_frequency)
                    t2s.append(1.0 / popt[4])
                    new_freqs.append(corrected_qubit_frequency)

                except Exception as e:
                    log.warning(f"ramsey_fit: the fitting was not succesful. {e}")
                    t2s.append(0)
                    deltas_phys_list.append(0)
                    new_freqs.append(0)

                freq_av[qubit] = np.mean(new_freqs)
                freq_err[qubit] = np.std(new_freqs)
                t2_av[qubit] = np.mean(t2s)
                t2_err[qubit] = np.std(t2s)
                delta_phys_av[qubit] = np.mean(deltas_phys_list)
                delta_phys_err[qubit] = np.std(deltas_phys_list)

        msrs = np.mean(msrs, axis=1).flatten()

        try:
            popts[qubit] = fitting(waits, msrs)

        except Exception as e:
            log.warning(f"ramsey_fit: the fitting was not succesful. {e}")
            popts[qubit] = [0] * 5

        if data.nboot == 0:
            delta_fitting = popts[qubit][2] / (2 * np.pi)
            delta_phys = data.detuning_sign * int(
                (delta_fitting - data.n_osc / data.t_max) * GHZ_TO_HZ
            )
            # FIXME: for qblox the correct formula is the following (there is a bug related to the phase)
            # corrected_qubit_frequency = int(qubit_freq + delta_phys)
            corrected_qubit_frequency = int(qubit_freq - delta_phys)
            t2 = 1.0 / popts[qubit][4]

            freq_av[qubit] = corrected_qubit_frequency
            freq_err[qubit] = 0.0
            t2_av[qubit] = t2
            t2_err[qubit] = 0.0
            delta_phys_av[qubit] = delta_phys
            delta_phys_err[qubit] = 0.0

    return RamseyResults(
        freq_av,
        freq_err,
        t2_av,
        t2_err,
        delta_phys_av,
        delta_phys_err,
        popts,
    )


def _plot(data: RamseyData, fit: RamseyResults, qubit):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    qubit_data = data.data[qubit]
    waits = np.unique(data.data[qubit].wait)
    msrs = np.array(qubit_data[["msr"]].tolist()).flatten() * V_TO_UV
    if data.nboot != 0:
        msrs = msrs.reshape((len(waits), -1))
        nsamples = msrs.shape[1]
        error_bars = np.std(msrs, axis=1).flatten() / np.sqrt(nsamples)
        msrs = np.mean(msrs, axis=1).flatten()
    else:
        error_bars = np.zeros(len(msrs))
    fig.add_trace(
        go.Scatter(
            x=waits,
            y=msrs,
            error_y=dict(
                type="data",  # value of error bar given in data coordinates
                array=error_bars,
                visible=True,
            ),
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        )
    )

    # add fitting trace
    waitrange = np.linspace(
        min(waits),
        max(waits),
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
        + (
            f"{qubit} | Delta_frequency: {fit.delta_phys[qubit]:,.1f} {chr(177)} {fit.error_delta_phys[qubit]:,.1f} Hz<br>"
        )
        + (
            f"{qubit} | Drive_frequency: {fit.frequency[qubit] } {chr(177)} {fit.error_frequency[qubit]:,.0f} Hz<br>"
        )
        + (
            f"{qubit} | T2*: {fit.t2[qubit]:,.0f} {chr(177)} {fit.error_t2[qubit]:,.0f} ns.<br><br>"
        )
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


def fitting(x, y):
    y_max = np.max(y)
    y_min = np.min(y)
    y = (y - y_min) / (y_max - y_min)
    x_max = np.max(x)
    x_min = np.min(x)
    x = (x - x_min) / (x_max - x_min)

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
    return popt
