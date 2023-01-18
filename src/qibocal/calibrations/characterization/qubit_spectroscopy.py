import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Sweeper

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Qubit Drive Frequency", plots.frequency_msr_phase)
def qubit_spectroscopy(
    platform: AbstractPlatform,
    qubits: list,
    fast_width,
    fast_step,
    precision_width,
    precision_step,
    wait_time,
    nshots=1024,
    software_averages=1,
):
    r"""
    Perform spectroscopy on the qubit.
    This routine executes a fast scan around the expected qubit frequency indicated in the platform runcard.
    Afterthat, a final sweep with more precision is executed centered in the new qubit frequency found.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        fast_start (int): Initial frequency in HZ to perform the qubit fast sweep
        fast_width (int): Width frequency in HZ to perform the high resolution sweep
        fast_step (int): Step frequency in HZ for the high resolution sweep
        precision_width (int): Width frequency in HZ to perform the precision resolution sweep
        precision_step (int): Step frequency in HZ for the precission resolution sweep
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - Two DataUnits objects with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Qubit drive frequency value in Hz
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **qubit**: The qubit being tested
            - **qubit_freq**: frequency
            - **peak_voltage**: peak voltage
            - **popt0**: Lorentzian's amplitude
            - **popt1**: Lorentzian's center
            - **popt2**: Lorentzian's sigma
            - **popt3**: Lorentzian's offset
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=2000
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    delta_frequency_range = np.arange(-fast_width // 2, fast_width // 2, fast_step)
    sweeper = Sweeper(
        "frequency",
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
        wait_time=wait_time,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit frequency
    fast_sweep_data = DataUnits(
        name="fast_sweep_data",
        quantities={"frequency": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        results = platform.sweep(sequence, sweeper, nshots=nshots)

        while (
            any(result.in_progress for result in results.values())
            or len(fast_sweep_data) == 0
        ):
            # retrieve the results for every qubit
            for qubit, ro_pulse in ro_pulses.items():
                # average msr, phase, i and q over the number of shots defined in the runcard
                result = results[ro_pulse.serial]
                # store the results
                r = {
                    "MSR[V]": result.MSR,
                    "i[V]": result.I,
                    "q[V]": result.Q,
                    "phase[rad]": result.phase,
                    "frequency[Hz]": delta_frequency_range + qd_pulses[qubit].frequency,
                    "qubit": len(result) * [qubit],
                    "iteration": len(result) * [iteration],
                }
                fast_sweep_data.add_data_from_dict(r)

            # save data as often as defined by points
            if len(result) > 0:
                # save data
                yield fast_sweep_data
                # calculate and save fit
                yield lorentzian_fit(
                    fast_sweep_data,
                    x="frequency[Hz]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    labels=["qubit_freq", "peak_voltage", "intermediate_freq"],
                )

    # finally, save the remaining data and fits
    yield fast_sweep_data
    yield lorentzian_fit(
        fast_sweep_data,
        x="frequency[Hz]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        labels=["qubit_freq", "peak_voltage", "intermediate_freq"],
    )

    # store max/min peaks as new frequencies
    # new_qubit_frequencies = {}
    # for qubit in qubits:
    #    qubit_data = (
    #        fast_sweep_data.df[fast_sweep_data.df["qubit"] == qubit]
    #        .drop(columns=["qubit", "iteration"])
    #        .groupby("frequency", as_index=False)
    #        .mean()
    #    )
    #    if platform.resonator_type == "3D":
    #        new_qubit_frequencies[qubit] = (
    #            qubit_data["frequency"][
    #                np.argmin(qubit_data["MSR"].pint.to("V").pint.magnitude)
    #            ]
    #            .to("Hz")
    #            .magnitude
    #        )
    #    else:
    #        new_qubit_frequencies[qubit] = (
    #            qubit_data["frequency"][
    #                np.argmax(qubit_data["MSR"].pint.to("V").pint.magnitude)
    #            ]
    #            .to("Hz")
    #            .magnitude
    #        )

    # run a precision sweep around the newly detected frequencies

    delta_frequency_range = np.arange(
        -precision_width // 2, precision_width // 2, precision_step
    )
    sweeper = Sweeper(
        "frequency",
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
        wait_time=wait_time,
    )

    # create a second DataUnits object to store the results,
    precision_sweep_data = DataUnits(
        name="precision_sweep_data",
        quantities={"frequency": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        results = platform.sweep(sequence, sweeper, nshots=nshots)

        while (
            any(result.in_progress for result in results.values())
            or len(precision_sweep_data) == 0
        ):
            # retrieve the results for every qubit
            for qubit, ro_pulse in ro_pulses.items():
                # average msr, phase, i and q over the number of shots defined in the runcard
                result = results[ro_pulse.serial]
                # store the results
                r = {
                    "MSR[V]": result.MSR,
                    "i[V]": result.I,
                    "q[V]": result.Q,
                    "phase[rad]": result.phase,
                    "frequency[Hz]": delta_frequency_range + qd_pulses[qubit].frequency,
                    "qubit": len(result) * [qubit],
                    "iteration": len(result) * [iteration],
                }
                precision_sweep_data.add_data_from_dict(r)

            # save data as often as defined by points
            if len(result) > 0:
                # save data
                yield precision_sweep_data
                # calculate and save fit
                yield lorentzian_fit(
                    precision_sweep_data,
                    x="frequency[Hz]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    labels=["qubit_freq", "peak_voltage", "intermediate_freq"],
                )

    # finally, save the remaining data and fits
    yield precision_sweep_data
    yield lorentzian_fit(
        precision_sweep_data,
        x="frequency[Hz]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        labels=["qubit_freq", "peak_voltage", "intermediate_freq"],
    )


@plot(
    "MSR and Phase vs Qubit Drive Frequency and Flux Current",
    plots.frequency_flux_msr_phase,
)
def qubit_spectroscopy_flux(
    platform: AbstractPlatform,
    qubits: list,
    freq_width,
    freq_step,
    bias_width,
    bias_step,
    fluxlines,
    wait_time,
    nshots=1024,
    software_averages=1,
):
    r"""
    Perform spectroscopy on the qubit modifying the current applied in the flux control line.
    This routine works for multiqubit devices flux controlled.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        freq_width (int): Width frequency in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequency in HZ for the spectroscopy sweep
        current_width (float): Width current in A for the flux current sweep
        current_step (float): Step current in A for the flux current sweep
        fluxlines (list): List of flux lines to use to perform the experiment. If it is set to "qubits", it uses each of
                        flux lines associated with the target qubits.
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Qubit drive frequency value in Hz
            - **current[A]**: Current value in A applied to the flux line
            - **qubit**: The qubit being tested
            - **fluxline**: The fluxline being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **qubit_freq**: frequency
            - **peak_voltage**: peak voltage
            - **popt0**: Lorentzian's amplitude
            - **popt1**: Lorentzian's center
            - **popt2**: Lorentzian's sigma
            - **popt3**: Lorentzian's offset
            - **qubit**: The qubit being tested
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=2000
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive frequency
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
    frequency_sweeper = Sweeper(
        "frequency",
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
        wait_time=wait_time,
    )

    if fluxlines == "qubits":
        fluxlines = qubits

    # flux current
    delta_bias_range = np.arange(-bias_width / 2, bias_width / 2, bias_step)
    bias_sweeper = Sweeper("offset", delta_bias_range, qubits=fluxlines, wait_time=0)

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit frequency and flux current
    data = DataUnits(
        name=f"data",
        quantities={"frequency": "Hz", "current": "A"},
        options=["qubit", "fluxline", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(software_averages):
        results = platform.sweep(
            sequence, bias_sweeper, frequency_sweeper, nshots=nshots
        )

        while any(result.in_progress for result in results.values()) or len(data) == 0:
            # retrieve the results for every qubit
            for qubit, fluxline in zip(qubits, fluxlines):
                # average msr, phase, i and q over the number of shots defined in the runcard
                result = results[ro_pulses[qubit].serial]
                # store the results
                biases = (
                    np.repeat(delta_bias_range, len(delta_frequency_range))
                    + platform.qubits[fluxline].flux.offset
                )
                freqs = np.array(
                    len(delta_bias_range)
                    * list(delta_frequency_range + qd_pulses[qubit].frequency)
                ).flatten()
                r = result.to_dict()
                r.update(
                    {
                        "frequency[Hz]": freqs,
                        "current[A]": biases,
                        "qubit": len(result) * [qubit],
                        "fluxline": len(result) * [fluxline],
                        "iteration": len(result) * [iteration],
                    }
                )
                data.add_data_from_dict(r)

            # save data as often as defined by points
            if len(result) > 0:
                # save data
                yield data

    # finally, save the remaining data and fits
    yield data
