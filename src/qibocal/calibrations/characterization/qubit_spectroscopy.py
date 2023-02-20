import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.config import raise_error
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Qubit Drive Frequency", plots.frequency_msr_phase)
def qubit_spectroscopy(
    platform: AbstractPlatform,
    qubits: dict,
    fast_width,
    fast_step,
    precision_width,
    precision_step,
    software_averages=1,
    points=10,
):
    r"""
    Perform spectroscopy on the qubit.
    This routine executes a fast scan around the expected qubit frequency indicated in the platform runcard.
    Afterthat, a final sweep with more precision is executed centered in the new qubit frequency found.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
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
            - **drive_frequency**: frequency
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
                    labels=["drive_frequency", "peak_voltage"],
                )
            # reconfigure the instruments based on the new resonator frequency
            # in this case setting the local oscillators
            # the pulse sequence does not need to be modified or recreated between executions
            for qubit in qubits:
                qd_pulses[qubit].frequency = delta_freq + qubits[qubit].drive_frequency

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            # retrieve the results for every qubit
            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].to_dict()
                # store the results
                r.update(
                    {
                        "frequency[Hz]": qd_pulses[ro_pulse.qubit].frequency,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                    }
                )
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
        labels=["drive_frequency", "peak_voltage"],
    )

    # store max/min peaks as new frequencies
    for qubit in qubits:
        qubit_data = (
            fast_sweep_data.df[fast_sweep_data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby("frequency", as_index=False)
            .mean()
        )
        if platform.resonator_type == "3D":
            qubits[qubit].drive_frequency = (
                qubit_data["frequency"][
                    np.argmin(qubit_data["MSR"].pint.to("V").pint.magnitude)
                ]
                .to("Hz")
                .magnitude
            )
        else:
            qubits[qubit].drive_frequency = (
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
                    labels=["resonator_freq", "peak_voltage"],
                )
            # reconfigure the instrument based on the new resonator frequency
            # in this case setting the local oscillators
            # the pulse sequence does not need to be modified between executions
            for qubit in qubits:
                qd_pulses[qubit].frequency = delta_freq + qubits[qubit].drive_frequency

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            # retrieve the results for every qubit
            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].to_dict()
                # store the results
                r.update(
                    {
                        "frequency[Hz]": qd_pulses[ro_pulse.qubit].frequency,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                    }
                )
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
        labels=["drive_frequency", "peak_voltage"],
    )


@plot(
    "MSR and Phase vs Qubit Drive Frequency and Flux Current",
    plots.frequency_flux_msr_phase,
)
def qubit_spectroscopy_flux(
    platform: AbstractPlatform,
    qubits: dict,
    freq_width,
    freq_step,
    bias_width,
    bias_step,
    fluxlines,
    software_averages=1,
    points=10,
):
    r"""
    Perform spectroscopy on the qubit modifying the bias applied in the flux control line.
    This routine works for multiqubit devices flux controlled.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        freq_width (int): Width frequency in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequency in HZ for the spectroscopy sweep
        bias_width (float): Width bias in A for the flux bias sweep
        bias_step (float): Step bias in A for the flux bias sweep
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
            - **bias[V]**: Current value in A applied to the flux line
            - **qubit**: The qubit being tested
            - **fluxline**: The fluxline being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **drive_frequency**: frequency
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
            qubit, start=0, duration=5000
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=5000)
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive frequency
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)

    # flux bias
    sweetspot_biass = {}
    bias_ranges = {}
    bias_min = {}
    bias_max = {}

    if fluxlines == "qubits":
        fluxlines = qubits

    for fluxline in fluxlines:
        # TODO: check if this is correct
        sweetspot_biass[fluxline] = qubits[fluxline].sweetspot

        bias_min[fluxline] = max(-bias_width / 2 + sweetspot_biass[fluxline], -0.03)
        bias_max[fluxline] = min(+bias_width / 2 + sweetspot_biass[fluxline], +0.03)
        bias_ranges[fluxline] = np.arange(
            bias_min[fluxline], bias_max[fluxline], bias_step
        )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit frequency and flux bias
    data = DataUnits(
        name=f"data",
        quantities={"frequency": "Hz", "bias": "V"},
        options=["qubit", "fluxline", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(software_averages):
        # sweep the parameters
        for fluxline in fluxlines:
            for bias in bias_ranges[fluxline]:
                # set new flux bias
                platform.set_bias(fluxline, bias)

                # TODO: adjust resonator frequency if coefs available in the runcard
                # coefs should be determined in resonator_spectroscopy_flux
                # matrix of biass -> magnetic flux -> freq shift

                for delta_freq in delta_frequency_range:
                    # save data as often as defined by points
                    if count % points == 0:
                        # save data
                        yield data
                        # TODO: calculate and save fit

                    # set new lo frequency
                    for qubit in qubits:
                        qd_pulses[qubit].frequency = (
                            delta_freq + qubits[qubit].drive_frequency
                        )

                    # execute the pulse sequence
                    result = platform.execute_pulse_sequence(sequence)

                    # retrieve the results for every qubit
                    for ro_pulse in ro_pulses.values():
                        # average msr, phase, i and q over the number of shots defined in the runcard
                        r = result[ro_pulse.serial].to_dict()
                        # store the results
                        r.update(
                            {
                                "frequency[Hz]": qd_pulses[ro_pulse.qubit].frequency,
                                "bias[V]": bias,
                                "qubit": ro_pulse.qubit,
                                "fluxline": fluxline,
                                "iteration": iteration,
                            }
                        )
                        data.add(r)
                    count += 1
    # finally, save the remaining data and fits
    yield data
