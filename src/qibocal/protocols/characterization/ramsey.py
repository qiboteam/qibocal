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
from scipy.signal import find_peaks

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.fitting.classifier import qubit_fit

from .utils import GHZ_TO_HZ


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

    frequency: dict[QubitId, tuple[float, Optional[float]]] = field(
        metadata=dict(update="drive_frequency")
    )
    """Drive frequency [GHz] for each qubit."""
    t2: dict[QubitId, tuple[float, Optional[float]]]
    """T2 for each qubit [ns]."""
    delta_phys: dict[QubitId, tuple[float, Optional[float]]]
    """Drive frequency [Hz] correction for each qubit."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""
    chi2: dict[QubitId, float]


RamseyType = np.dtype(
    [("wait", np.float64), ("prob", np.float64), ("errors", np.float64)]
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

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, wait, prob, errors):
        """Store output for single qubit."""
        # to be able to handle the non-sweeper case
        ar = np.empty(prob.shape, dtype=RamseyType)
        ar["wait"] = np.array([wait])
        ar["prob"] = np.array([prob])

        ar["errors"] = np.array([errors])

        if qubit in self.data:
            # print(type(self.data[qubit]["prob"]), type(ar))
            # print(self.data[qubit])
            # self.data[qubit] = np.rec.array(self.data[qubit]["wait"])
            # l = np.concatenate((self.data[qubit]["prob"], ar["prob"]))
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
            prob = []  # TODO: evaluate prob
            data.register_qubit(qubit, wait=waits, prob=prob)

    else:
        probs = [[] for _ in qubits]
        errors = [[] for _ in qubits]
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
                    averaging_mode=(
                        AveragingMode.SINGLESHOT
                        # if params.nboot != 0
                        # else AveragingMode.CYCLIC
                    ),
                ),
            )
            # print("PROVA")
            print("DDDD ", probs)
            for i, qubit in enumerate(qubits):
                result = results[ro_pulses[qubit].serial]
                i_values = result.voltage_i
                # print(len(i_values))
                q_values = result.voltage_q
                iq_couples = np.stack((i_values, q_values), axis=-1)
                # print(iq_couples)
                model = qubit_fit.QubitFit()
                model.angle = platform.qubits[qubit].iq_angle
                model.threshold = platform.qubits[qubit].threshold
                model.iq_mean0 = platform.qubits[qubit].mean_gnd_states
                model.iq_mean1 = platform.qubits[qubit].mean_exc_states
                states = model.predict(iq_couples)
                # print(states)
                # states = np.reshape(states, (len(waits), -1))
                prob = np.average(
                    states,
                )
                number_ones = np.sum(states)
                print(probs)
                errors[i].append(
                    np.sqrt(number_ones * (len(states) - number_ones) / len(states))
                )
                print(probs)
                # print(errors[i])
                print(probs, prob)
                probs[i].append(prob)
    # print(errors)
    for i, qubit in enumerate(qubits):
        print(len(probs[i]), len(errors[i]), len(waits))
        print(errors[i])
        data.register_qubit(
            qubit,
            wait=np.array(waits),
            prob=np.array(probs[i]),
            errors=np.array(errors[i]),
        )
    # print(data)
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
    waits = data.waits
    popts = {}
    freq_measure = {}
    t2_measure = {}
    delta_phys_measure = {}
    chi2 = {}
    for qubit in qubits:
        qubit_data = data[qubit]
        qubit_freq = data.qubit_freqs[qubit]
        probs = qubit_data[["prob"]].tolist()
        # t2s = []
        # deltas_phys_list = []
        # new_freqs = []

        #     nsamples = probs.shape[1]
        #     # error_bars = np.std(probs, axis=1).flatten() / np.sqrt(nsamples)
        # bootstrap_samples = bootstrap(probs, data.nboot)
        # voltages = np.mean(bootstrap_samples, axis=1)
        # fit_out = []
        # for i in range(data.nboot):
        #     y = voltages[:, i]
        #     x = waits
        #     try:
        #         popt = fitting(x, y)

        #         delta_fitting = popt[2] / (2 * np.pi)
        #         delta_phys = data.detuning_sign * int(
        #             (delta_fitting - data.n_osc / data.t_max) * GHZ_TO_HZ
        #         )

        #         corrected_qubit_frequency = int(qubit_freq - delta_phys)
        #         deltas_phys_list.append(delta_phys)
        #         new_freqs.append(corrected_qubit_frequency)
        #         t2s.append(1.0 / popt[4])
        #         new_freqs.append(corrected_qubit_frequency)

        #     except Exception as e:
        #         popt = [0] * 5
        #         log.warning(f"ramsey_fit: the fitting was not succesful. {e}")
        #         t2s.append(0)
        #         deltas_phys_list.append(0)
        #         new_freqs.append(0)

        #     fit_out.append(popt)
        #     freq_measure[qubit] = (np.mean(new_freqs), np.std(new_freqs))
        #     t2_measure[qubit] = (np.mean(t2s), np.std(t2s))
        #     delta_phys_measure[qubit] = (
        #         np.mean(deltas_phys_list),
        #         np.std(deltas_phys_list),
        #     )
        #     popts[qubit] = np.mean(fit_out, axis=0).tolist()
        #     y = np.array(y)
        #     x = np.array(x)
        #     chi2[qubit] = chi2_reduced(y, ramsey_fit(x, *popts[qubit]), error_bars)

        # else:
        # import matplotlib.pyplot as plt
        # plt.scatter(waits, prob)
        # plt.savefig("ramsey.png")
        try:
            popt = fitting(waits, probs)
        except:
            popt = [0, 0, 0, 0, 1]
        delta_fitting = popt[2] / (2 * np.pi)
        delta_phys = data.detuning_sign * int(
            (delta_fitting - data.n_osc / data.t_max) * GHZ_TO_HZ
        )
        corrected_qubit_frequency = int(qubit_freq - delta_phys)
        t2 = 1.0 / popt[4]

        freq_measure[qubit] = (corrected_qubit_frequency, None)
        t2_measure[qubit] = (t2, None)
        popts[qubit] = popt
        delta_phys_measure[qubit] = delta_phys
        chi2[qubit] = [0, 0]  # TODO:remove
    return RamseyResults(freq_measure, t2_measure, delta_phys_measure, popts, chi2)


def _plot(data: RamseyData, qubit, fit: RamseyResults = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = None

    qubit_data = data.data[qubit]
    waits = data.waits
    probs = qubit_data[["prob"]].tolist()
    # if data.nboot != 0:
    #     # print(probs.shape)
    #     probs = np.reshape(probs, (len(waits), -1))
    #     nsamples = probs.shape[1]
    #     error_bars = np.std(probs, axis=1).flatten() / np.sqrt(nsamples)
    #     probs = np.mean(probs, axis=1).flatten()
    # else:
    probs = np.reshape(probs, (len(waits)))
    error_bars = qubit_data["errors"]

    fig.add_trace(
        go.Scatter(
            x=waits,
            y=probs,
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

    if fit is not None:
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
        fitting_report = ""
        # fitting_report = (
        #     ""
        #     + (
        #         fill_table(
        #             qubit,
        #             "Delta_frequency",
        #             fit.delta_phys[qubit][0],
        #             fit.delta_phys[qubit][1],
        #             "Hz",
        #         )
        #     )
        #     + (
        #         fill_table(
        #             qubit,
        #             "Drive_frequency",
        #             fit.frequency[qubit][0],
        #             fit.frequency[qubit][1],
        #             "Hz",
        #         )
        #     )
        #     + (fill_table(qubit, "T2*", fit.t2[qubit][0], fit.t2[qubit][1], "ns"))
        #     + "<br>"
        # )
        # if fit.chi2:
        #     fitting_report += fill_table(
        #         qubit, "chi2 reduced", fit.chi2[qubit], error=None
        #     )
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


def fitting(x: list, y: list) -> list:
    """
    Given the inputs list `x` and outputs one `y`, this function fits the
    `ramsey_fit` function and returns a list with the fit parameters.
    """
    y_max = np.max(y)
    y_min = np.min(y)
    y = (y - y_min) / (y_max - y_min)
    x_max = np.max(x)
    x_min = np.min(x)
    x = (x - x_min) / (x_max - x_min)

    ft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), x[1] - x[0])
    mags = abs(ft)
    local_maxima = find_peaks(mags, threshold=10)[0]
    index = local_maxima[0] if len(local_maxima) > 0 else None
    # 0.5 hardcoded guess for less than one oscillation
    f = freqs[index] * 2 * np.pi if index is not None else 0.5
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
        maxfev=5000,
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
