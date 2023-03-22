import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Qubit Drive Frequency", plots.frequency_msr_phase)
def coupler_spectroscopy(
    platform: AbstractPlatform,
    qubits: dict,  # Better than giving the coupler to select the driving qubit and readout qubit
    coupler_frequency,
    frequency_width,
    frequency_step,
    coupler_drive_duration,
    qubit_drive_duration,
    coupler_drive_amplitude=None,
    qubit_drive_amplitude=None,
    nshots=1024,
    relaxation_time=50,
    software_averages=1,
):
    r"""
    Perform spectroscopy on the coupler.
    This routine executes a scan around the expected coupler frequency indicated in the platform runcard.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the driving action. Readout qubit is obtained from it as well.
        frequency_width (int): Width frequency in HZ to perform the high resolution sweep
        frequency_step (int): Step frequency in HZ for the high resolution sweep
        drive_duration,
        drive_amplitude=None,
        nshots=1024,
        relaxation_time=50,
        software_averages (int): Number of executions of the routine for averaging results

    Returns:
        - DataUnits object with the raw data obtained for the sweep with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Qubit drive frequency value in Hz
            - **coupler**: The coupler being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **coupler**: The qubit being tested
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

    # long weak drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    cd_pulses = {}
    # for qubit in qubits:

    qubit_drive = qubits[4].name
    qubit_readout = qubits[2].name  # Make general

    cd_pulses[qubit_drive] = platform.create_qubit_drive_pulse(
        qubit_drive, start=0, duration=coupler_drive_duration
    )
    cd_pulses[qubit_drive].frequency = coupler_frequency
    if coupler_drive_amplitude is not None:
        cd_pulses[qubit_drive].amplitude = coupler_drive_amplitude

    qd_pulses[qubit_readout] = platform.create_qubit_drive_pulse(
        qubit_readout,
        start=cd_pulses[qubit_drive].finish - 2000,
        duration=qubit_drive_duration,
    )

    if qubit_drive_amplitude is not None:
        qd_pulses[qubit_readout].amplitude = qubit_drive_amplitude

    ro_pulses[qubit_readout] = platform.create_qubit_readout_pulse(
        qubit_readout, start=qd_pulses[qubit_readout].finish
    )

    sequence.add(qd_pulses[qubit_readout])
    sequence.add(cd_pulses[qubit_drive])
    sequence.add(ro_pulses[qubit_readout])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -frequency_width // 2, frequency_width // 2, frequency_step
    )

    # sweeper = Sweeper(
    #     Parameter.frequency,
    #     delta_frequency_range,
    #     pulses=[qd_pulses[qubit] for qubit in qubits],
    # )

    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[cd_pulses[qubit_drive]],
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit frequency
    sweep_data = DataUnits(
        name="fast_sweep_data",
        quantities={"frequency": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        results = platform.sweep(
            sequence, sweeper, nshots=nshots, relaxation_time=relaxation_time
        )

        # retrieve the results for every qubit
        # for qubit, ro_pulse in ro_pulses.items():
        # average msr, phase, i and q over the number of shots defined in the runcard

        ro_pulse = ro_pulses[qubit_readout]
        result = results[ro_pulse.serial]
        r = result.to_dict(average=False)
        # store the results
        r.update(
            {
                "frequency[Hz]": delta_frequency_range
                + cd_pulses[qubit_drive].frequency,
                "qubit": len(delta_frequency_range) * [qubit_drive],
                "iteration": len(delta_frequency_range) * [iteration],
            }
        )
        sweep_data.add_data_from_dict(r)

    yield sweep_data
    # calculate and save fit
    # yield lorentzian_fit(
    #     sweep_data,
    #     x="frequency[Hz]",
    #     y="MSR[uV]",
    #     qubits=qubits,
    #     resonator_type=platform.resonator_type,
    #     labels=["drive_frequency", "peak_voltage"],
    # )


@plot("MSR and Phase vs Qubit Drive Frequency", plots.coupler_frequencies_msr_phase)
def coupler_spectroscopy_double_freq(
    platform: AbstractPlatform,
    qubits: dict,  # Better than giving the coupler to select the driving qubit and readout qubit
    coupler_frequency,
    frequency_coupler_width,
    frequency_coupler_step,
    frequency_drive_width,
    frequency_drive_step,
    coupler_drive_duration,
    qubit_drive_duration,
    coupler_drive_amplitude=None,
    qubit_drive_amplitude=None,
    nshots=1024,
    relaxation_time=50,
    software_averages=1,
):
    r"""
    Perform spectroscopy on the coupler.
    This routine executes a scan around the expected coupler frequency indicated in the platform runcard.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the driving action. Readout qubit is obtained from it as well.
        frequency_width (int): Width frequency in HZ to perform the high resolution sweep
        frequency_step (int): Step frequency in HZ for the high resolution sweep
        drive_duration,
        drive_amplitude=None,
        nshots=1024,
        relaxation_time=50,
        software_averages (int): Number of executions of the routine for averaging results

    Returns:
        - DataUnits object with the raw data obtained for the sweep with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Qubit drive frequency value in Hz
            - **coupler**: The coupler being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **coupler**: The qubit being tested
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

    # long weak drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    cd_pulses = {}
    # for qubit in qubits:

    qubit_drive = qubits[4].name
    # qubit_drive = qubits[0].name
    qubit_readout = qubits[2].name  # Make general

    cd_pulses[qubit_drive] = platform.create_qubit_drive_pulse(
        qubit_drive, start=0, duration=coupler_drive_duration
    )
    cd_pulses[qubit_drive].frequency = coupler_frequency
    if coupler_drive_amplitude is not None:
        cd_pulses[qubit_drive].amplitude = coupler_drive_amplitude

    qd_pulses[qubit_readout] = platform.create_qubit_drive_pulse(
        qubit_readout,
        start=cd_pulses[qubit_drive].finish,
        duration=qubit_drive_duration,
    )

    if qubit_drive_amplitude is not None:
        qd_pulses[qubit_readout].amplitude = qubit_drive_amplitude

    ro_pulses[qubit_readout] = platform.create_qubit_readout_pulse(
        qubit_readout, start=qd_pulses[qubit_readout].finish
    )

    sequence.add(qd_pulses[qubit_readout])
    sequence.add(cd_pulses[qubit_drive])
    sequence.add(ro_pulses[qubit_readout])

    # define the parameter to sweep and its range:
    delta_coupler_frequency_range = np.arange(
        -frequency_coupler_width // 2,
        frequency_coupler_width // 2,
        frequency_coupler_step,
    )

    delta_drive_frequency_range = np.arange(
        -frequency_drive_width // 2, frequency_drive_width // 4, frequency_drive_step
    )

    sweeper_coupler = Sweeper(
        Parameter.frequency,
        delta_coupler_frequency_range,
        pulses=[cd_pulses[qubit_drive]],
    )

    # sweeper_drive = Sweeper(
    #     Parameter.frequency,
    #     delta_drive_frequency_range,
    #     pulses=[qd_pulses[qubit_readout]],
    # )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit frequency
    sweep_data = DataUnits(
        name="sweep_data",
        quantities={"frequency_coupler": "Hz", "frequency_drive": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        og_freq = qd_pulses[qubit_readout].frequency
        for freq in delta_drive_frequency_range:
            # platform.connect()

            platform.qubits[2].drive_frequency = og_freq + freq
            qd_pulses[qubit_readout].frequency = og_freq + freq

            results = platform.sweep(
                sequence,
                sweeper_coupler,
                nshots=nshots,
                relaxation_time=relaxation_time,
            )

            # retrieve the results for every qubit
            # for qubit, ro_pulse in ro_pulses.items():
            # average msr, phase, i and q over the number of shots defined in the runcard

            ro_pulse = ro_pulses[qubit_readout]
            result = results[ro_pulse.serial]
            r = result.to_dict(average=False)
            # store the results
            r.update(
                {
                    "frequency_coupler[Hz]": delta_coupler_frequency_range
                    + cd_pulses[qubit_drive].frequency,
                    # "frequency_drive[Hz]": delta_drive_frequency_range
                    # + qd_pulses[qubit_readout].frequency,
                    "frequency_drive[Hz]": len(delta_coupler_frequency_range)
                    * [qd_pulses[qubit_readout].frequency],
                    "qubit": len(delta_coupler_frequency_range) * [qubit_drive],
                    "iteration": len(delta_coupler_frequency_range) * [iteration],
                }
            )
            sweep_data.add_data_from_dict(r)
            # platform.disconnect()

    yield sweep_data
    # calculate and save fit
    # yield lorentzian_fit(
    #     sweep_data,
    #     x="frequency[Hz]",
    #     y="MSR[uV]",
    #     qubits=qubits,
    #     resonator_type=platform.resonator_type,
    #     labels=["drive_frequency", "peak_voltage"],
    # )


@plot("MSR and Phase vs Qubit Drive Frequency", plots.coupler_frequency_flux_msr_phase)
def coupler_spectroscopy_flux(
    platform: AbstractPlatform,
    qubits: dict,  # Better than giving the coupler to select the driving qubit and readout qubit
    coupler_frequency,
    coupler_sweetspot,
    frequency_width,
    frequency_step,
    coupler_drive_duration,
    qubit_drive_duration,
    coupler_drive_amplitude=None,
    qubit_drive_amplitude=None,
    bias_width=None,
    bias_step=None,
    coupler_line=None,
    nshots=1024,
    relaxation_time=50,
    software_averages=1,
):
    r"""
    Perform spectroscopy on the coupler.
    This routine executes a scan around the expected coupler frequency indicated in the platform runcard.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the driving action. Readout qubit is obtained from it as well.
        frequency_width (int): Width frequency in HZ to perform the high resolution sweep
        frequency_step (int): Step frequency in HZ for the high resolution sweep
        drive_duration,
        drive_amplitude=None,
        nshots=1024,
        relaxation_time=50,
        software_averages (int): Number of executions of the routine for averaging results

    Returns:
        - DataUnits object with the raw data obtained for the sweep with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Qubit drive frequency value in Hz
            - **coupler**: The coupler being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **coupler**: The qubit being tested
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

    # long weak drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    cd_pulses = {}
    # for qubit in qubits:

    qubit_drive = qubits[4].name
    qubit_readout = qubits[2].name  # Make general

    qd_pulses[qubit_readout] = platform.create_qubit_drive_pulse(
        qubit_readout, start=500, duration=qubit_drive_duration
    )

    if qubit_drive_amplitude is not None:
        qd_pulses[qubit_readout].amplitude = qubit_drive_amplitude

    cd_pulses[qubit_drive] = platform.create_qubit_drive_pulse(
        qubit_drive, start=0, duration=coupler_drive_duration
    )
    cd_pulses[qubit_drive].frequency = coupler_frequency
    if coupler_drive_amplitude is not None:
        cd_pulses[qubit_drive].amplitude = coupler_drive_amplitude

    ro_pulses[qubit_readout] = platform.create_qubit_readout_pulse(
        qubit_readout, start=qd_pulses[qubit_readout].finish
    )

    sequence.add(qd_pulses[qubit_readout])
    sequence.add(cd_pulses[qubit_drive])
    sequence.add(ro_pulses[qubit_readout])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -frequency_width // 2, frequency_width // 4, frequency_step
    )

    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[cd_pulses[qubit_drive]],
    )

    if coupler_line == "qubits":
        fluxlines = qubits
    if coupler_line == "c4":
        fluxlines = {}
        fluxlines[0] = qubits["c4"]
        sweetspot = coupler_sweetspot

    # flux bias
    delta_bias_range = np.arange(-bias_width / 2, bias_width / 2, bias_step) + sweetspot
    bias_sweeper = Sweeper(
        Parameter.bias,
        delta_bias_range,
        qubits=fluxlines,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit frequency
    sweep_data = DataUnits(
        name="data",
        quantities={"frequency": "Hz", "bias": "dimensionless"},
        options=["qubit", "fluxline", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        results = platform.sweep(
            sequence,
            bias_sweeper,
            sweeper,
            nshots=nshots,
            relaxation_time=relaxation_time,
        )

        # retrieve the results for every qubit
        # for qubit, ro_pulse in ro_pulses.items():
        # average msr, phase, i and q over the number of shots defined in the runcard

        ro_pulse = ro_pulses[qubit_readout]
        result = results[ro_pulse.serial]

        # Check this for how to do the sweeps
        biases = np.array(len(delta_frequency_range) * list(delta_bias_range)).flatten()
        # ) + platform.get_bias(fluxline)
        freqs = np.repeat(
            delta_frequency_range + cd_pulses[qubit_drive].frequency,
            len(delta_bias_range),
        )

        r = {k: v.ravel() for k, v in result.to_dict(average=False).items()}
        # store the results
        r.update(
            {
                "frequency[Hz]": freqs,
                "bias[dimensionless]": biases,
                "qubit": len(freqs) * [qubit_readout],
                "fluxline": len(freqs) * [coupler_line],
                "iteration": len(freqs) * [iteration],
            }
        )
        sweep_data.add_data_from_dict(r)

    yield sweep_data
    # calculate and save fit
    # yield lorentzian_fit(
    #     sweep_data,
    #     x="frequency[Hz]",
    #     y="MSR[uV]",
    #     qubits=qubits,
    #     resonator_type=platform.resonator_type,
    #     labels=["drive_frequency", "peak_voltage"],
    # )
