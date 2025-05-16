from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import mpld3
import numpy as np
import numpy.typing as npt
from qibolab import Platform
from qm import generate_qua_script, qua
from qm.qua import (
    align,
    assign,
    case_,
    declare,
    declare_stream,
    dual_demod,
    else_,
    fixed,
    for_,
    for_each_,
    if_,
    measure,
    play,
    save,
    switch_,
    wait,
)
from qualang_tools.bakery import baking
from qualang_tools.results import fetching_tool
from scipy import signal
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.protocols.utils import table_dict, table_html

from .configuration import generate_config


@dataclass
class CryoscopeQuaParameters(Parameters):
    flux_amplitude: float
    const_flux_len: int = 200
    zeros_before_pulse: int = 20
    """Beginning of the flux pulse (before we put zeros to see the rising time)."""
    zeros_after_pulse: int = 20
    """End of the flux pulse (after we put zeros to see the falling time)"""
    other_qubits: list[QubitId] = field(default_factory=list)
    """Qubits to set the bias offset to zero (parking)."""
    debug: Optional[str] = None
    "If enabled it dumps the qua script in a file with the given name."


CryoscopeQuaType = np.dtype(
    [
        ("state", float),
    ]
)


@dataclass
class CryoscopeQuaData(Data):
    const_flux_len: int
    zeros_before_pulse: int
    zeros_after_pulse: int
    sampling_rate: int
    data: dict[QubitId, npt.NDArray[CryoscopeQuaType]] = field(default_factory=dict)


def baked_waveform(waveform, pulse_duration, config, flux, sampling_rate):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(
            config, padding_method="right", sampling_rate=sampling_rate * 1e9
        ) as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * (16 * sampling_rate)
            else:
                wf = waveform[:i].tolist()

            b.add_op("flux_pulse", flux, wf)
            b.play("flux_pulse", flux)
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


def _acquisition(
    params: CryoscopeQuaParameters, platform: Platform, targets: list[QubitId]
) -> CryoscopeQuaData:
    assert len(targets) == 1
    target = targets[0]

    res_name = platform.qubits[target].acquisition
    qbit_name = platform.qubits[target].drive
    flux = platform.qubits[target].flux
    # This threshold is further towards the ground IQ blob to increase the initialization fidelity
    # initialization_threshold = -0.00025
    flux_config = platform.config(flux)
    sampling_rate = (
        int(flux_config.sampling_rate / 1e9)
        if hasattr(flux_config, "sampling_rate")
        else 1
    )

    # FLux pulse waveform generation
    flux_amp = params.flux_amplitude
    const_flux_len = sampling_rate * params.const_flux_len
    zeros_before_pulse = sampling_rate * params.zeros_before_pulse
    zeros_after_pulse = sampling_rate * params.zeros_after_pulse

    flux_pulse = np.array([flux_amp] * const_flux_len)
    flux_waveform = np.array(
        [0.0] * zeros_before_pulse + list(flux_pulse) + [0.0] * zeros_after_pulse
    )
    total_len = const_flux_len + zeros_before_pulse + zeros_after_pulse

    # Baked flux pulse segments
    config = generate_config(platform, platform.qubits.keys())
    square_pulse_segments = baked_waveform(
        flux_waveform, total_len, config, flux, sampling_rate
    )

    acquisition_config = platform.config(platform.qubits[target].acquisition)
    threshold = acquisition_config.threshold
    cos_iq = np.cos(acquisition_config.iq_angle)
    sin_iq = np.sin(acquisition_config.iq_angle)

    with qua.program() as cryoscope:
        n = declare(int)  # Variable for averaging
        n_st = declare_stream()
        signal_i = declare(fixed)  # I quadrature for state measurement
        signal_q = declare(fixed)  # Q quadrature for state measurement
        state = declare(bool)  # Qubit state
        state_st = declare_stream()
        # I_g = declare(fixed)  # I quadrature for qubit cooldown
        segment = declare(int)  # Flux pulse segment
        flag = declare(
            bool
        )  # Boolean flag to switch between x90 and y90 for state measurement

        # Set the flux line offset of the other qubit to 0
        # qua.set_dc_offset(flux, "single", 0)
        for q in params.other_qubits:
            qua.set_dc_offset(f"flux{q}", "single", 0)

        with for_(n, 0, n < params.nshots, n + 1):
            # Notice it's <= to include t_max (This is only for integers!)
            with for_(segment, 0, segment <= total_len, segment + 1):
                with for_each_(flag, [True, False]):
                    # Cooldown
                    # measure("readout", res_name, None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
                    # with while_(I_g > initialization_threshold):
                    #    measure("readout", res_name, None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
                    align()
                    # wait(500)
                    wait(params.relaxation_time // 4)
                    # Cryoscope protocol
                    # Play the first pi/2 pulse
                    play("x90", qbit_name)
                    align(qbit_name, flux)
                    # Play truncated flux pulse with 1ns resolution
                    with switch_(segment):
                        for j in range(0, total_len + 1):
                            with case_(j):
                                square_pulse_segments[j].run()
                    # Wait some fixed time so that the whole protocol duration is constant
                    wait(total_len // (4 * sampling_rate), qbit_name)
                    # Play the second pi/2 pulse along x and y successively
                    with if_(flag):
                        play("x90", qbit_name)
                    with else_():
                        play("y90", qbit_name)
                    # State readout
                    align(qbit_name, res_name)
                    measure(
                        "measure",
                        res_name,
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", signal_i),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", signal_q),
                        # dual_demod.full("rotated_cos", "rotated_sin", I),
                        # dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
                    )
                    # State discrimination
                    # assign(state, I > ge_threshold)
                    assign(state, signal_i * cos_iq - signal_q * sin_iq > threshold)
                    save(state, state_st)
            save(n, n_st)

        with qua.stream_processing():
            state_st.boolean_to_int().buffer(2).buffer(total_len + 1).average().save(
                "state"
            )
            n_st.save("iteration")

    controller = platform._controller
    qmm = controller.manager
    # Open the quantum machine
    qm = qmm.open_qm(config)

    if params.debug is not None:
        with open(params.debug, "w") as file:
            file.write(generate_qua_script(cryoscope, config))

    # Execute QUA program
    job = qm.execute(cryoscope)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["state", "iteration"], mode="live")
    while results.is_processing():
        # Fetch results
        state, iteration = results.fetch_all()

    # At the end of the program, fetch the non-averaged results to get the error-bars
    results = fetching_tool(job, data_list=["state"])
    state = results.fetch_all()[0]

    data = CryoscopeQuaData(
        const_flux_len=const_flux_len,
        zeros_before_pulse=zeros_before_pulse,
        zeros_after_pulse=zeros_after_pulse,
        sampling_rate=sampling_rate,
    )
    data.register_qubit(CryoscopeQuaType, target, {"state": state})
    return data


# Exponential decay
def expdecay(x, a, t):
    """Exponential decay defined as 1 + a * np.exp(-x / t).
    :param x: numpy array for the time vector in ns
    :param a: float for the exponential amplitude
    :param t: float for the exponential decay time in ns
    :return: numpy array for the exponential decay
    """
    return 1 + a * np.exp(-x / t)


# Theoretical IIR and FIR taps based on exponential decay coefficients
def exponential_correction(A, tau, Ts=1e-9):
    """Derive FIR and IIR filter taps based on a the exponential coefficients A and tau from 1 + a * np.exp(-x / t).
    :param A: amplitude of the exponential decay
    :param tau: decay time of the exponential decay
    :param Ts: sampling period. Default is 1e-9
    :return: FIR and IIR taps
    """
    tau = tau * Ts
    k1 = Ts + 2 * tau * (A + 1)
    k2 = Ts - 2 * tau * (A + 1)
    c1 = Ts + 2 * tau
    c2 = Ts - 2 * tau
    feedback_tap = -k2 / k1
    feedforward_taps = np.array([c1, c2]) / k1
    return feedforward_taps, feedback_tap


# FIR and IIR taps calculation
def filter_calc(exponential):
    """Derive FIR and IIR filter taps based on a list of exponential coefficients.
    :param exponential: exponential coefficients defined as [(A1, tau1), (A2, tau2)]
    :return: FIR and IIR taps as [fir], [iir]
    """
    # Initialization based on the number of exponential coefficients
    b = np.zeros((2, len(exponential)))
    feedback_taps = np.zeros(len(exponential))
    # Derive feedback tap for each set of exponential coefficients
    for i, (A, tau) in enumerate(exponential):
        b[:, i], feedback_taps[i] = exponential_correction(A, tau)
    # Derive feddback tap for each set of exponential coefficients
    feedforward_taps = b[:, 0]
    for i in range(len(exponential) - 1):
        feedforward_taps = np.convolve(feedforward_taps, b[:, i + 1])
    # feedforward taps are bounded to +/- 2
    if np.abs(max(feedforward_taps)) >= 2:
        feedforward_taps = 2 * feedforward_taps / max(feedforward_taps)

    return feedforward_taps, feedback_taps


@dataclass
class CryoscopeQuaResults(Results):
    alpha: dict[QubitId, float] = field(default_factory=dict)
    tau: dict[QubitId, float] = field(default_factory=dict)
    fir: dict[QubitId, float] = field(default_factory=dict)
    iir: dict[QubitId, float] = field(default_factory=dict)


def calculate_step_response(data: CryoscopeQuaData, qubit: QubitId) -> npt.NDArray:
    state = data[qubit]["state"]
    Sxx = state[:, 0] * 2 - 1  # Bloch vector projection along X
    Syy = state[:, 1] * 2 - 1  # Bloch vector projection along Y
    S = Sxx + 1j * Syy  # Bloch vector
    # Qubit phase
    phase = np.unwrap(np.angle(S))
    phase = phase - phase[-1]
    # Qubit detuning
    detuning = signal.savgol_filter(
        phase[data.zeros_before_pulse : data.const_flux_len + data.zeros_before_pulse]
        / 2
        / np.pi,
        21,
        2,
        deriv=1,
        delta=0.001,
    )
    # Step response
    return np.sqrt(detuning / np.average(detuning[-int(data.const_flux_len / 4)]))


def _fit(data: CryoscopeQuaData) -> CryoscopeQuaResults:
    results = CryoscopeQuaResults()
    for qubit in data.qubits:
        const_flux_len = data.const_flux_len
        step_response_volt = calculate_step_response(data, qubit)

        ## Fit step response with exponential
        [A, tau], _ = curve_fit(
            expdecay,
            np.arange(0, const_flux_len, 1) / data.sampling_rate,
            # xplot[zeros_before_pulse : const_flux_len + zeros_before_pulse],
            step_response_volt,
        )
        results.alpha[qubit] = A
        results.tau[qubit] = tau
        ## Derive IIR and FIR corrections
        fir, iir = filter_calc(exponential=[(A, tau)])
        results.fir[qubit] = list(fir)
        results.iir[qubit] = list(iir)
    return results


def _plot(data: CryoscopeQuaData, target: QubitId, fit: CryoscopeQuaResults):
    const_flux_len = data.const_flux_len
    # xplot = np.arange(0, data.total_len + 0.1, 1)[
    #    zeros_before_pulse : const_flux_len + zeros_before_pulse
    # ]
    xplot = np.arange(0, const_flux_len, 1) / data.sampling_rate
    A = fit.alpha[target]
    tau = fit.tau[target]
    fir = np.array(fit.fir[target])
    iir = np.array(fit.iir[target])
    step_response_volt = calculate_step_response(data, target)

    ## Derive responses and plots
    # Ideal response
    pulse = np.array([1.0] * const_flux_len)
    # Response without filter
    no_filter = expdecay(xplot, a=A, t=tau)
    # Response with filters
    with_filter = no_filter * signal.lfilter(
        fir, [1, iir[0]], pulse
    )  # Output filter , DAC Output

    fitting_report = table_html(
        table_dict(
            target,
            ["A", "tau", "FIR", "IIR"],
            [
                (A,),
                (tau,),
                (fir,),
                (iir,),
            ],
        )
    )

    # Plot all data
    fig = plt.figure(figsize=(20, 6))
    plt.rcParams.update({"font.size": 13})
    plt.suptitle("Cryoscope with filter implementation")
    plt.subplot(121)
    plt.plot(xplot, step_response_volt, "o-", label="Data")
    plt.plot(xplot, expdecay(xplot, A, tau), label="Fit")
    # plt.axhline(y=1.01)
    # plt.axhline(y=0.99)
    plt.xlabel("Flux pulse duration [ns]")
    plt.ylabel("Step response")
    plt.legend()

    plt.subplot(122)
    plt.plot()
    plt.plot(no_filter, label="After Bias-T without filter")
    plt.plot(with_filter, label="After Bias-T with filter")
    plt.plot(pulse, label="Ideal WF")  # pulse
    plt.plot(list(step_response_volt), label="Experimental data")
    plt.xlabel("Flux pulse duration [ns]")
    plt.ylabel("Step response")
    plt.legend(loc="upper right")

    figures = [mpld3.fig_to_html(fig)]
    return figures, fitting_report


def _update(results: CryoscopeQuaResults, platform: Platform, target: QubitId):
    pass


qua_cryoscope = Routine(_acquisition, _fit, _plot, _update)
