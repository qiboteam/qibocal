import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

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
    software_averages,
    points=10,
):
    r"""
    Perform spectroscopy on the qubit.
    This routine executes a fast scan around the expected qubit frequency indicated in the platform runcard.
    Afterthat, a final sweep with more precision is executed centered in the new qubit frequency found.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        fast_start (int): Initial frequenecy in HZ to perform the qubit fast sweep
        fast_end (int): End frequenecy in HZ to perform the qubit fast sweep
        fast_step (int): Step frequenecy in HZ for the qubit fast sweep
        precision_start (int): Initial frequenecy in HZ to perform the qubit precision sweep
        precision_end (int): End frequenecy in HZ to perform the qubit precision sweep
        precision_step (int): Step frequenecy in HZ for the qubit precision sweep
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz

        - A DataUnits object with the fitted data obtained with the following keys

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
            qubit, start=0, duration=5000
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=5000)
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive frequency
    qubit_frequencies = {}
    for qubit in qubits:
        qubit_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "qubit_freq"
        ]
    delta_frequency_range = np.arange(-fast_width // 2, fast_width // 2, fast_step)

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit frequency
    fast_sweep_data = DataUnits(
        name="fast_sweep_data",
        quantities={"frequency": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(software_averages):
        # sweep the parameter
        for delta_freq in delta_frequency_range:
            # save data as often as defined by points
            if count % points == 0 and count > 0:
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
            # reconfigure the instruments based on the new resonator frequency
            # in this case setting the local oscillators
            # the pulse sequence does not need to be modified or recreated between executions
            for qubit in qubits:
                platform.qd_port[qubit].lo_frequency = (
                    delta_freq + qubit_frequencies[qubit] - qd_pulses[qubit].frequency
                )

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            # retrieve the results for every qubit
            for qubit in qubits:
                # average msr, phase, i and q over the number of shots defined in the runcard
                msr, phase, i, q = results[ro_pulses[qubit].serial]
                # store the results
                r = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": delta_freq + qubit_frequencies[qubit],
                    "qubit": qubit,
                    "iteration": iteration,
                }
                fast_sweep_data.add(r)
            count += 1
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
    new_qubit_frequencies = {}
    for qubit in qubits:
        qubit_data = (
            fast_sweep_data.df[fast_sweep_data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby("frequency", as_index=False)
            .mean()
        )
        if platform.resonator_type == "3D":
            new_qubit_frequencies[qubit] = (
                qubit_data["frequency"][
                    np.argmin(qubit_data["MSR"].pint.to("V").pint.magnitude)
                ]
                .to("Hz")
                .magnitude
            )
        else:
            new_qubit_frequencies[qubit] = (
                qubit_data["frequency"][
                    np.argmax(qubit_data["MSR"].pint.to("V").pint.magnitude)
                ]
                .to("Hz")
                .magnitude
            )

    # run a precision sweep around the newly detected frequencies

    delta_frequency_range = np.arange(
        -precision_width // 2, precision_width // 2, precision_step
    )

    # create a second DataUnits object to store the results,
    precision_sweep_data = DataUnits(
        name="precision_sweep_data",
        quantities={"frequency": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(software_averages):
        # sweep the parameter
        for delta_freq in delta_frequency_range:
            # save data as often as defined by points
            if count % points == 0 and count > 0:
                # save data
                yield precision_sweep_data
                # calculate and save fit
                yield lorentzian_fit(
                    precision_sweep_data,
                    x="frequency[Hz]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    labels=["resonator_freq", "peak_voltage", "intermediate_freq"],
                )
            # reconfigure the instrument based on the new resonator frequency
            # in this case setting the local oscillators
            # the pulse sequence does not need to be modified between executions
            for qubit in qubits:
                platform.qd_port[qubit].lo_frequency = (
                    delta_freq
                    + new_qubit_frequencies[qubit]
                    - qd_pulses[qubit].frequency
                )

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            # retrieve the results for every qubit
            for pulse in sequence.ro_pulses:
                # average msr, phase, i and q over the number of shots defined in the runcard
                msr, phase, i, q = results[pulse.serial]
                # store the results
                r = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": delta_freq + new_qubit_frequencies[pulse.qubit],
                    "qubit": pulse.qubit,
                    "iteration": iteration,
                }
                precision_sweep_data.add(r)
            count += 1
    # finally, save the remaining data and fits
    yield precision_sweep_data
    yield lorentzian_fit(
        precision_sweep_data,
        x="frequency[Hz]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        labels=["resonator_freq", "peak_voltage", "intermediate_freq"],
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
    current_width,
    current_step,
    software_averages,
    fluxlines,
    points=10,
):
    r"""
    Perform spectroscopy on the qubit modifying the current applied in the flux control line.
    This routine works for multiqubit devices flux controlled.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        freq_width (int): Width frequenecy in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequenecy in HZ for the spectroscopy sweep
        current_max (int): Minimum value in mV for the flux current sweep
        current_min (int): Minimum value in mV for the flux current sweep
        current_step (int): Step attenuation in mV for the flux current sweep
        software_averages (int): Number of executions of the routine for averaging results
        fluxline (int): Flux line associated to the target qubit. If it is set to "qubit", the platform
                automatically obtain the flux line number of the target qubit.
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz

        - A DataUnits object with the fitted data obtained with the following keys

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
            qubit, start=0, duration=5000
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=5000)
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive frequency
    qubit_frequencies = {}
    for qubit in qubits:
        qubit_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "qubit_freq"
        ]
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)

    # flux current
    sweetspot_currents = {}
    current_ranges = {}
    current_min = {}
    current_max = {}

    if fluxlines == "qubits":
        fluxlines = qubits

    for fluxline in fluxlines:
        sweetspot_currents[fluxline] = platform.characterization["single_qubit"][qubit][
            "sweetspot"
        ]
        current_min[fluxline] = max(
            -current_width / 2 + sweetspot_currents[fluxline], -0.03
        )
        current_max[fluxline] = min(
            +current_width / 2 + sweetspot_currents[fluxline], +0.03
        )
        current_ranges[fluxline] = np.arange(
            current_min[fluxline], current_max[fluxline], current_step
        )

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
        # sweep the parameters
        for fluxline in fluxlines:
            for current in current_ranges[fluxline]:
                # set new flux current
                platform.qf_port[fluxline].current = current

                # TODO: adjust resonator frequency if coefs available in the runcard
                # coefs should be determined in resonator_spectroscopy_flux
                # matrix of currents -> magnetic flux -> freq shift

                for delta_freq in delta_frequency_range:
                    # save data as often as defined by points
                    if count % points == 0:
                        # save data
                        yield data
                        # TODO: calculate and save fit

                    # set new lo frequency
                    for qubit in qubits:
                        platform.qd_port[qubit].lo_frequency = (
                            delta_freq
                            + qubit_frequencies[qubit]
                            - qd_pulses[qubit].frequency
                        )

                    # execute the pulse sequence
                    result = platform.execute_pulse_sequence(sequence)

                    # retrieve the results for every qubit
                    for qubit in qubits:
                        # average msr, phase, i and q over the number of shots defined in the runcard
                        msr, phase, i, q = result[ro_pulses[qubit].serial]
                        # store the results
                        r = {
                            "MSR[V]": msr,
                            "i[V]": i,
                            "q[V]": q,
                            "phase[rad]": phase,
                            "frequency[Hz]": delta_freq + qubit_frequencies[qubit],
                            "current[A]": current,
                            "qubit": qubit,
                            "fluxline": fluxline,
                            "iteration": iteration,
                        }
                        data.add(r)
                    count += 1
    # finally, save the remaining data and fits
    yield data
