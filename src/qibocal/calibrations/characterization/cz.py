import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, PulseType, Rectangular

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("Chevron CZ", plots.duration_amplitude_msr_flux_pulse)
def tune_transition(
    platform: AbstractPlatform,
    qubits: list,
    flux_pulse_duration_start,
    flux_pulse_duration_end,
    flux_pulse_duration_step,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
    wait_time,
    single_flux=True,
    dt=1,
    nshots=1024,
):
    """Perform a Chevron-style plot for the flux pulse designed to apply a CZ (CPhase) gate.
    This experiment probes the |11> to i|02> transition by preparing the |11> state with
    pi-pulses, applying a flux pulse to the high frequency qubit to engage its 1 -> 2 transition
    with varying interaction duration and amplitude. We then measure both the high and low frequency qubit.

    We aim to find the spot where the transition goes from |11> -> i|02> -> -|11>.

    Args:
        platform: platform where the experiment is meant to be run.
        qubit (int): qubit that will interact with center qubit 2.
        flux_pulse_duration_start (int): minimum flux pulse duration in nanoseconds.
        flux_pulse_duration_end (int): maximum flux pulse duration in nanoseconds.
        flux_pulse_duration_step (int): step for the duration sweep in nanoseconds.
        flux_pulse_amplitude_start (float): minimum flux pulse amplitude.
        flux_pulse_amplitude_end (float): maximum flux pulse amplitude.
        flux_pulse_amplitude_step (float): step for the amplitude sweep.
        single_flux (bool): use a single pulse or two flux pulses with half duration and opposite amplitude.
        dt (int): time delay between the two flux pulses if enabled.

    Returns:
        data (DataSet): Measurement data for both the high and low frequency qubits.

    """
    if len(qubits) > 1:
        raise NotImplementedError

    qubit = qubits[0]

    platform.reload_settings()

    initialize_1 = platform.create_RX_pulse(qubit, start=0, relative_phase=0)
    initialize_2 = platform.create_RX_pulse(2, start=0, relative_phase=0)

    # TODO: generalize this to take any two qubits that are connected to the chip.
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    if single_flux:
        flux_pulse = FluxPulse(
            start=initialize_1.se_finish,
            duration=flux_pulse_duration_start,
            amplitude=flux_pulse_amplitude_start,
            relative_phase=0,
            shape=Rectangular(),
            channel=str(platform.qubits[highfreq].flux),
            qubit=highfreq,
        )
        measure_lowfreq = platform.create_qubit_readout_pulse(
            lowfreq, start=flux_pulse.se_finish
        )
        measure_highfreq = platform.create_qubit_readout_pulse(
            highfreq, start=flux_pulse.se_finish
        )

    else:
        raise NotImplementedError
        flux_pulse_plus = FluxPulse(
            start=initialize_1.se_finish,
            duration=flux_pulse_duration_start,
            amplitude=flux_pulse_amplitude_start,
            relative_phase=0,
            shape=Rectangular(),
            channel=str(platform.qubits[highfreq].flux),
            qubit=highfreq,
        )
        flux_pulse_minus = FluxPulse(
            start=flux_pulse_plus.se_finish + dt,
            duration=flux_pulse_duration_start,
            amplitude=-flux_pulse_amplitude_start,
            relative_phase=0,
            shape=Rectangular(),
            channel=str(platform.qubits[highfreq].flux),
            qubit=highfreq,
        )
        measure_lowfreq = platform.create_qubit_readout_pulse(
            lowfreq, start=flux_pulse_minus.se_finish
        )
        measure_highfreq = platform.create_qubit_readout_pulse(
            highfreq, start=flux_pulse_minus.se_finish
        )

    data = DataUnits(
        name=f"data_q{lowfreq}{highfreq}",
        quantities={
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
        options=["q_freq"],
    )

    amplitudes = np.arange(
        flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
    )
    durations = np.arange(
        flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
    )
    # TODO: Implement for two pulses
    sweeper = Sweeper("amplitude", amplitudes, pulses=[flux_pulse], wait_time=wait_time)

    if single_flux:
        sequence = (
            initialize_1
            + initialize_2
            + flux_pulse
            + measure_lowfreq
            + measure_highfreq
        )
    else:
        sequence = (
            initialize_1
            + initialize_2
            + flux_pulse_plus
            + flux_pulse_minus
            + measure_lowfreq
            + measure_highfreq
        )

    # Might want to fix duration to expected time for 2 qubit gate.
    for duration in durations:
        if single_flux:
            flux_pulse.amplitude = amplitude
            flux_pulse.duration = duration
        else:
            flux_pulse_plus.amplitude = amplitude
            flux_pulse_minus.amplitude = -amplitude
            flux_pulse_plus.duration = duration
            flux_pulse_minus.duration = duration

        results = platform.sweep(sequence, sweeper, nshots=nshots)

        res_temp = results[measure_lowfreq.serial].to_dict()
        res_temp.update(
            {
                "flux_pulse_duration[ns]": len(amplitudes) * [duration],
                "flux_pulse_amplitude[dimensionless]": amplitudes,
                "q_freq": len(amplitudes) * ["low"],
            }
        )
        data.add_data_from_dict(res_temp)

        res_temp = results[measure_highfreq.serial].to_dict()
        res_temp.update(
            {
                "flux_pulse_duration[ns]": len(amplitudes) * [duration],
                "flux_pulse_amplitude[dimensionless]": amplitudes,
                "q_freq": len(amplitudes) * ["high"],
            }
        )
        data.add_data_from_dict(res_temp)

    yield data


@plot("Landscape 2-qubit gate", plots.landscape_2q_gate)
def tune_landscape(
    platform: AbstractPlatform,
    qubit: int,
    theta_start,
    theta_end,
    theta_step,
    flux_pulse_duration,
    flux_pulse_amplitude,
    single_flux=True,
    dt=1,
):
    """Check the two-qubit landscape created by a flux pulse of a given duration
    and amplitude.
    The system is initialized with a Y90 pulse on the low frequency qubit and either
    an Id or an X gate on the high frequency qubit. Then the flux pulse is applied to
    the high frequency qubit in order to perform a two-qubit interaction. The Id/X gate
    is undone in the high frequency qubit and a theta90 pulse is applied to the low
    frequency qubit before measurement. That is, a pi-half pulse around the relative phase
    parametereized by the angle theta.

    Measurements on the low frequency qubit yield the the 2Q-phase of the gate and the
    remnant single qubit Z phase aquired during the execution to be corrected.
    Population of the high frequency qubit yield the leakage to the non-computational states
    during the execution of the flux pulse.

    Args:
        platform: platform where the experiment is meant to be run.
        qubit (int): qubit that will interact with center qubit 2.
        theta_start (float): initial angle for the low frequency qubit measurement in radians.
        theta_end (float): final angle for the low frequency qubit measurement in radians.
        theta_step, (float): step size for the theta sweep in radians.
        flux_pulse_duration (int): fixed duration for the flux pulse sent to the high frequency qubit.
        flux_pulse_amplitude (float): fixed amplitude for the flux pulse sent to the high frequency qubit.
        single_flux (bool): use a single pulse or two flux pulses with half duration and opposite amplitude.
        dt (int): time delay between the two flux pulses if enabled.

    Returns:
        data (DataSet): Measurement data for both the high and low frequency qubits for the two setups of Id/X.

    """

    platform.reload_settings()

    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    x_pulse_start = platform.create_RX_pulse(highfreq, start=0, relative_phase=0)
    y90_pulse = platform.create_RX90_pulse(lowfreq, start=0, relative_phase=np.pi / 2)

    if single_flux:
        flux_pulse = FluxPulse(
            start=y90_pulse.se_finish,
            duration=flux_pulse_duration,
            amplitude=flux_pulse_amplitude,
            relative_phase=0,
            shape=Rectangular(),
            channel=platform.qubit_channel_map[highfreq][2],
            qubit=highfreq,
        )
        theta_pulse = platform.create_RX90_pulse(
            lowfreq, flux_pulse.se_finish, relative_phase=theta_start
        )
        x_pulse_end = platform.create_RX_pulse(
            highfreq, start=flux_pulse.se_finish, relative_phase=0
        )

    else:
        flux_pulse_plus = FluxPulse(
            start=y90_pulse.se_finish,
            duration=flux_pulse_duration,
            amplitude=flux_pulse_amplitude,
            relative_phase=0,
            shape=Rectangular(),
            channel=platform.qubit_channel_map[highfreq][2],
            qubit=highfreq,
        )
        flux_pulse_minus = FluxPulse(
            start=flux_pulse_plus.se_finish + dt,
            duration=flux_pulse_duration,
            amplitude=-flux_pulse_amplitude,
            relative_phase=0,
            shape=Rectangular(),
            channel=platform.qubit_channel_map[highfreq][2],
            qubit=highfreq,
        )
        theta_pulse = platform.create_RX90_pulse(
            lowfreq, flux_pulse_minus.se_finish, relative_phase=theta_start
        )
        x_pulse_end = platform.create_RX_pulse(
            highfreq, start=flux_pulse_minus.se_finish, relative_phase=0
        )

    measure_lowfreq = platform.create_qubit_readout_pulse(
        lowfreq, start=theta_pulse.se_finish
    )
    measure_highfreq = platform.create_qubit_readout_pulse(
        highfreq, start=theta_pulse.se_finish
    )

    data = DataUnits(
        name=f"data_q{lowfreq}{highfreq}",
        quantities={
            "theta": "ns",
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
        options=["q_freq", "setup"],
    )

    thetas = np.arange(theta_start, theta_end, theta_step)

    setups = ["I", "X"]

    for setup in setups:
        if setup == "I":
            if single_flux:
                seq = (
                    y90_pulse
                    + flux_pulse
                    + theta_pulse
                    + measure_lowfreq
                    + measure_highfreq
                )
            else:
                seq = (
                    y90_pulse
                    + flux_pulse_plus
                    + flux_pulse_minus
                    + theta_pulse
                    + measure_lowfreq
                    + measure_highfreq
                )
        elif setup == "X":
            if single_flux:
                seq = (
                    x_pulse_start
                    + y90_pulse
                    + flux_pulse
                    + theta_pulse
                    + x_pulse_end
                    + measure_lowfreq
                    + measure_highfreq
                )
            else:
                seq = (
                    x_pulse_start
                    + y90_pulse
                    + flux_pulse_plus
                    + flux_pulse_minus
                    + theta_pulse
                    + x_pulse_end
                    + measure_lowfreq
                    + measure_highfreq
                )

        live = 0
        for theta in thetas:

            theta_pulse.relative_phase = theta

            res = platform.execute_pulse_sequence(seq)
            res_temp = res[measure_lowfreq.serial]
            results = {
                "MSR[V]": res_temp[0],
                "i[V]": res_temp[2],
                "q[V]": res_temp[3],
                "phase[rad]": res_temp[1],
                "flux_pulse_duration[ns]": flux_pulse_duration,
                "flux_pulse_amplitude[dimensionless]": flux_pulse_amplitude,
                "q_freq": "low",
                "setup": setup,
            }
            data.add(results)

            res_temp = res[measure_highfreq.serial]
            results = {
                "MSR[V]": res_temp[0],
                "i[V]": res_temp[2],
                "q[V]": res_temp[3],
                "phase[rad]": res_temp[1],
                "flux_pulse_duration[ns]": flux_pulse_duration,
                "flux_pulse_amplitude[dimensionless]": flux_pulse_amplitude,
                "q_freq": "high",
                "setup": setup,
            }
            data.add(results)

            if live % 10 == 0:
                yield data

            live += 1

    yield data
