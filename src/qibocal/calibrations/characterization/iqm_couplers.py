import itertools

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.platforms.platform import AcquisitionType, AveragingMode
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Qubit Drive Frequency", plots.coupler_frequency_msr_phase)
def coupler_spectroscopy(
    platform: AbstractPlatform,
    qubits: dict,
    coupler_frequency,
    frequency_width,
    frequency_step,
    coupler_drive_duration,
    coupler_drive_amplitude=None,
    qubit_drive_amplitude=None,
    nshots=1024,
    relaxation_time=50,
):
    r"""
    Perform spectroscopy on the coupler by 1D scan on the coupler driving pulse frequency with readout fixed on the qubit peak. https://arxiv.org/pdf/2208.09460.pdf

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the {coupler driving through this qubit, qubit driving + readout, coupling}.
        coupler_frequency (int): Frequency in HZ around which we will perform the sweep
        frequency_width (int): Width frequency in HZ to perform the sweep
        frequency_step (int): Step frequency in HZ for the sweep
        coupler_drive_duration (float) : Duration in ns of the coupler driving pulse
        qubit_drive_duration (float) : Duration in ns of the qubit driving pulse
        coupler_drive_amplitude (float) : Amplitude in A.U. of the coupler driving pulse
        qubit_drive_amplitude (float) : Amplitude in A.U. of the qubit driving pulse
        nshots (int) : Number of executions on hardware of the routine for averaging results
        relaxation_time (float): Wait time for the qubit to decohere back to the `gnd` state

    Returns:
        - DataUnits object with the raw data obtained for the sweep with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Coupler drive frequency value in Hz
            - **coupler**: The qubit which is being used to drive the coupler
            - **iteration**: The SINGLE software iteration number

        - A DataUnits object with the fitted data obtained with the following keys

            - **coupler**: The qubit which is being used to drive the coupler
            - **drive_frequency**: frequency
            - **peak_voltage**: peak voltage
            - **popt0**: Lorentzian's amplitude
            - **popt1**: Lorentzian's center
            - **popt2**: Lorentzian's sigma
            - **popt3**: Lorentzian's offset
    """

    # create a sequence of pulses for the experiment:
    # long weak drive probing pulse - MZ

    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    cd_pulses = {}
    # for qubit in qubits:
    qubit_drive = platform.qubits[qubits[0].name].name
    qubit_readout = platform.qubits[2].name  # Make general

    cd_pulses[qubit_drive] = platform.create_qubit_drive_pulse(
        qubit_drive, start=0, duration=coupler_drive_duration
    )

    platform.qubits[qubits[0].name].drive.local_oscillator.frequency = (
        coupler_frequency - 100_000_000
    )
    platform.qubits[qubits[0].name].drive_frequency = coupler_frequency
    cd_pulses[qubit_drive].frequency = coupler_frequency

    if coupler_drive_amplitude is not None:
        cd_pulses[qubit_drive].amplitude = coupler_drive_amplitude

    qd_pulses[qubit_readout] = platform.create_qubit_drive_pulse(
        qubit_readout,
        start=0,
        duration=coupler_drive_duration,
    )

    if qubit_drive_amplitude is not None:
        qd_pulses[qubit_readout].amplitude = qubit_drive_amplitude

    ro_pulses[qubit_readout] = platform.create_qubit_readout_pulse(
        qubit_readout, start=0  # qd_pulses[qubit_readout].finish
    )

    sequence.add(qd_pulses[qubit_readout])
    sequence.add(cd_pulses[qubit_drive])
    sequence.add(ro_pulses[qubit_readout])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -frequency_width // 2, frequency_width // 2, frequency_step
    )

    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[cd_pulses[qubit_drive]],
    )

    # create a DataUnits object to store the results,
    sweep_data = DataUnits(
        name="sweep_data",
        quantities={"frequency": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by nshots
    iteration = 0
    results = platform.sweep(
        sequence,
        sweeper,
        nshots=nshots,
        relaxation_time=relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # retrieve the results for every qubit
    ro_pulse = ro_pulses[qubit_readout]
    result = results[ro_pulse.serial]
    r = result.raw
    # store the results
    r.update(
        {
            "frequency[Hz]": delta_frequency_range + cd_pulses[qubit_drive].frequency,
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
    qubits: dict,
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
):
    r"""
    Perform spectroscopy on the coupler by 2D scan on the coupler driving pulse frequency alongisde qubit driving pulse frequency. https://arxiv.org/pdf/2208.09460.pdf
    This routine executes a scan around the qubit frequency indicated in the platform runcard.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the driving action. Readout qubit is obtained from it as well.
        coupler_frequency (int): Frequency in HZ around which we will perform the sweep
        frequency_coupler_width (int): Width frequency in HZ to perform the coupler sweep
        frequency_coupler_step (int): Step frequency in HZ for the coupler sweep
        frequency_drive_width (int): Width frequency in HZ to perform the qubit sweep
        frequency_drive_step (int): Step frequency in HZ for the qubit sweep
        coupler_drive_duration (float) : Duration in ns of the coupler driving pulse
        qubit_drive_duration (float) : Duration in ns of the qubit driving pulse
        coupler_drive_amplitude (float) : Amplitude in A.U. of the coupler driving pulse
        qubit_drive_amplitude (float) : Amplitude in A.U. of the qubit driving pulse
        nshots (int) : Number of executions on hardware of the routine for averaging results
        relaxation_time (float): Wait time for the qubit to decohere back to the `gnd` state
    Returns:
        - DataUnits object with the raw data obtained for the sweep with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency_coupler[Hz]**: Coupler drive frequency value in Hz
            - **frequency_drive[Hz]**: Qubit drive frequency value in Hz
            - **coupler**: The qubit which is being used to drive the coupler
            - **iteration**: The SINGLE software iteration number

        - A DataUnits object with the fitted data obtained with the following keys

            - **coupler**: The qubit being tested
            - **drive_frequency**: frequency
            - **peak_voltage**: peak voltage
            - **popt0**: Lorentzian's amplitude
            - **popt1**: Lorentzian's center
            - **popt2**: Lorentzian's sigma
            - **popt3**: Lorentzian's offset
    """

    # create a sequence of pulses for the experiment:
    # long weak drive probing pulse - MZ
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    cd_pulses = {}
    # for qubit in qubits:

    # FIXME: Frequency results may be wrong

    qubit_drive = platform.qubits[qubits[0]].name
    # qubit_drive = qubits[0].name
    qubit_readout = platform.qubits[2].name  # Make general

    cd_pulses[qubit_drive] = platform.create_qubit_drive_pulse(
        qubit_drive, start=0, duration=coupler_drive_duration
    )

    platform.qubits[qubits[0]].drive.local_oscillator.frequency = coupler_frequency
    platform.qubits[qubits[0]].drive_frequency = coupler_frequency
    cd_pulses[qubit_drive].frequency = coupler_frequency

    if coupler_drive_amplitude is not None:
        cd_pulses[qubit_drive].amplitude = coupler_drive_amplitude

    qd_pulses[qubit_readout] = platform.create_qubit_drive_pulse(
        qubit_readout,
        # start=cd_pulses[qubit_drive].finish,
        start=0,
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

    # Hardware frequency sweep may drive only a single oscillator (workaround ?)
    sweeper_drive = Sweeper(
        Parameter.frequency,
        delta_drive_frequency_range,
        pulses=[qd_pulses[qubit_readout]],
    )

    # create a DataUnits object to store the results,
    sweep_data = DataUnits(
        name="sweep_data",
        quantities={"frequency_coupler": "Hz", "frequency_drive": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by nshots
    iteration = [0]
    og_freq = qd_pulses[qubit_readout].frequency

    platform.qubits[2].drive_frequency = og_freq
    qd_pulses[qubit_readout].frequency = og_freq

    results = platform.sweep(
        sequence,
        sweeper_coupler,
        sweeper_drive,
        nshots=nshots,
        relaxation_time=relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # retrieve the results for every qubit
    ro_pulse = ro_pulses[qubit_readout]
    result = results[ro_pulse.serial]
    # r = result.to_dict(average=False)

    # store the results
    drive_freqs = (
        np.repeat(delta_drive_frequency_range, len(delta_coupler_frequency_range))
        + qd_pulses[qubit_readout].frequency
    )
    coupler_freqs = np.array(
        len(delta_drive_frequency_range)
        * list(delta_coupler_frequency_range + cd_pulses[qubit_drive].frequency)
    ).flatten()
    r = {k: v.ravel() for k, v in result.raw.items()}
    r.update(
        {
            "frequency_coupler[Hz]": coupler_freqs,
            "frequency_drive[Hz]": drive_freqs,
            "qubit": len(coupler_freqs) * [qubit_drive],
            "iteration": len(coupler_freqs) * [iteration],
        }
    )
    sweep_data.add_data_from_dict(r)
    # platform.disconnect()

    yield sweep_data

    # for freq in delta_drive_frequency_range:
    #     # platform.connect()

    #     platform.qubits[2].drive_frequency = og_freq + freq
    #     qd_pulses[qubit_readout].frequency = og_freq + freq

    #     results = platform.sweep(
    #         sequence,
    #         sweeper_coupler,
    #         nshots=nshots,
    #         relaxation_time=relaxation_time,
    #     )

    #     # retrieve the results for every qubit
    #     ro_pulse = ro_pulses[qubit_readout]
    #     result = results[ro_pulse.serial]
    #     r = result.to_dict(average=False)
    #     # store the results
    #     r.update(
    #         {
    #             "frequency_coupler[Hz]": delta_coupler_frequency_range
    #             + cd_pulses[qubit_drive].frequency,
    #             "frequency_drive[Hz]": len(delta_coupler_frequency_range)
    #             * [qd_pulses[qubit_readout].frequency],
    #             "qubit": len(delta_coupler_frequency_range) * [qubit_drive],
    #             "iteration": len(delta_coupler_frequency_range) * [iteration],
    #         }
    #     )
    #     sweep_data.add_data_from_dict(r)
    #     # platform.disconnect()

    # yield sweep_data

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
    qubits: dict,
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
):
    r"""
    Perform spectroscopy on the coupler.
    This routine executes a scan around the expected coupler frequency indicated in the platform runcard.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the driving action. Readout qubit is obtained from it as well.
        coupler_frequency (int): Frequency in HZ around which we will perform the frequency outer sweep
        coupler_sweetspot (float): Amplitude in A.U. around which we will perform amplitude inner the sweep for the flux biasing pulses.
        frequency_width (int): Width frequency in HZ to perform the frequency outer sweep
        frequency_step (int): Step frequency in HZ for the high frequency outer sweep
        coupler_drive_duration (float) : Duration in ns of the coupler driving pulse
        qubit_drive_duration (float) : Duration in ns of the qubit driving pulse
        coupler_drive_amplitude (float) : Amplitude in A.U. of the coupler driving pulse
        qubit_drive_amplitude (float) : Amplitude in A.U. of the qubit driving pulse
        bias_width (int): Width bias in A.U. to perform the amplitude inner sweep
        bias_step (int): Step amplitude in A.U. for the amplitude inner sweep
        coupler_line (str): Name of the coupler qubit
        nshots (int) : Number of executions on hardware of the routine for averaging results
        relaxation_time (float): Wait time for the qubit to decohere back to the `gnd` state

    Returns:
        - DataUnits object with the raw data obtained for the sweep with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Coupler drive frequency value in Hz
            - **bias[dimensionless]**: Coupler flux bias pulse amplitude value in A.U.
            - **coupler**: The qubit which is being used to drive the coupler
            - **fluxline**: The coupler being tested
            - **iteration**: The SINGLE software iteration number

        - A DataUnits object with the fitted data obtained with the following keys

            - **coupler**: The qubit being tested
            - **drive_frequency**: frequency
            - **peak_voltage**: peak voltage
            - **popt0**: Lorentzian's amplitude
            - **popt1**: Lorentzian's center
            - **popt2**: Lorentzian's sigma
            - **popt3**: Lorentzian's offset
    """

    # create a sequence of pulses for the experiment:
    # long weak drive probing pulse - MZ
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    cd_pulses = {}
    # for qubit in qubits:

    qubit_drive = platform.qubits[qubits[0].name].name
    qubit_readout = platform.qubits[2].name  # Make general

    qd_pulses[qubit_readout] = platform.create_qubit_drive_pulse(
        qubit_readout, start=0, duration=qubit_drive_duration
    )

    if qubit_drive_amplitude is not None:
        qd_pulses[qubit_readout].amplitude = qubit_drive_amplitude

    cd_pulses[qubit_drive] = platform.create_qubit_drive_pulse(
        qubit_drive, start=0, duration=coupler_drive_duration
    )

    platform.qubits[qubits[0].name].drive.local_oscillator.frequency = coupler_frequency
    platform.qubits[qubits[0].name].drive_frequency = coupler_frequency
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
        -frequency_width // 2, frequency_width // 2, frequency_step
    )

    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[cd_pulses[qubit_drive]],
    )

    # if coupler_line == "qubits":
    #     fluxlines = qubits
    # if coupler_line == f"c{qubits[0].name}":
    fluxlines = {}
    fluxlines[0] = platform.qubits[f"c{qubits[0].name}"]
    sweetspot = coupler_sweetspot

    # flux bias
    delta_bias_range = np.arange(-bias_width / 2, bias_width / 2, bias_step) + sweetspot
    bias_sweeper = Sweeper(
        Parameter.bias,
        delta_bias_range,
        qubits=fluxlines,
    )

    # create a DataUnits object to store the results,
    sweep_data = DataUnits(
        name="data",
        quantities={"frequency": "Hz", "bias": "dimensionless"},
        options=["qubit", "fluxline", "iteration"],
    )

    # repeat the experiment as many times as defined by nshots
    iteration = 0
    results = platform.sweep(
        sequence,
        bias_sweeper,
        sweeper,
        nshots=nshots,
        relaxation_time=relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
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

    r = {k: v.ravel() for k, v in result.raw.items()}
    # store the results
    r.update(
        {
            "frequency[Hz]": freqs,
            "bias[dimensionless]": biases,
            "qubit": len(freqs) * [qubit_readout],
            "fluxline": len(freqs) * ["c0"],
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


def iq_to_probability(i, q, mean_gnd, mean_exc):
    state = i + 1j * q
    state = state - mean_gnd
    mean_exc = mean_exc - mean_gnd
    state = state * np.exp(-1j * np.angle(mean_exc))
    mean_exc = mean_exc * np.exp(-1j * np.angle(mean_exc))
    return np.real(state) / np.real(mean_exc)


@plot("coupler_swap", plots.coupler_swap)
def coupler_swap(
    platform: AbstractPlatform,
    qubits: dict,
    coupler_frequency: float,
    coupler_drive_duration: float,
    coupler_drive_amplitude: float,
    frequency_width: float,
    frequency_step: float,
    nshots: int,
):
    r"""
    Perform a SWAP experiment between two qubits through the coupler by changing its frequency.

    Args:
        platform (AbstractPlatform): The quantum platform to run the experiment on.
        qubits (dict): The qubits to perform the experiment on.
        coupler_frequency (float): The frequency of the coupler.
        coupler_drive_duration (float): The duration of the coupler drive pulse.
        coupler_drive_amplitude (float): The amplitude of the coupler drive pulse.
        frequency_width (float): The width of the frequency sweep.
        frequency_step (float): The step size of the frequency sweep.
        nshots (int): The number of shots to average over.

    Returns:
        - DataUnits: The results of the experiment.
            - **frequency[Hz]**: The frequency of the coupler.
            - **coupler** (str): The name of the coupler.
            - **qubit** (str): The name of the qubit measured.
            - **probability** (float): The probability of the qubit being in the excited state.
            - **state** (str): The state of the qubit.
    """
    # define the qubit to measure
    # FIXME: make general for multiple qubits
    if 2 in qubits:
        qubits.pop(2)
    qubit_pairs = list([qubit, 2] for qubit in qubits)

    # create a sequence
    sequence = PulseSequence()
    ro_pulses = {}
    cd_pulses = {}
    qd_pulses = {}

    for pair in qubit_pairs:
        qd_pulses[pair[1]] = platform.create_RX_pulse(pair[1], start=0)
        cd_pulses[pair[0]] = platform.create_qubit_drive_pulse(
            pair[0], start=qd_pulses[pair[1]].se_finish, duration=coupler_drive_duration
        )
        cd_pulses[pair[0]].amplitude = coupler_drive_amplitude
        platform.qubits[pair[0]].drive_frequency = coupler_frequency
        platform.qubits[pair[0]].drive.local_oscillator.frequency = (
            coupler_frequency - 100_000_000
        )

        ro_pulses[pair[1]] = platform.create_MZ_pulse(
            pair[1], start=cd_pulses[pair[0]].se_finish
        )
        # ro_pulses[pair[0]] = platform.create_MZ_pulse(pair[0], start=ro_pulses[pair[1]].se_finish) # Multiplex not working yet

        sequence.add(qd_pulses[pair[1]])
        sequence.add(cd_pulses[pair[0]])
        sequence.add(ro_pulses[pair[1]])
        # sequence.add(ro_pulses[pair[0]])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -frequency_width // 2, frequency_width // 2, frequency_step
    )

    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[cd_pulses[pair[0]] for pair in qubit_pairs],
    )

    # create a DataUnits object to store the results,
    sweep_data = DataUnits(
        name="data",
        quantities={"frequency": "Hz"},
        options=["coupler", "qubit", "state", "probability"],
    )

    # repeat the experiment as many times as defined by nshots
    results = platform.sweep(
        sequence,
        sweeper,
        nshots=nshots,
        acquisition_type=AcquisitionType.INTEGRATION,  # AcquisitionType.DISCRIMINATION
        averaging_mode=AveragingMode.CYCLIC,  # AcquisitionMode.CYCLIC
        relaxation_time=300e-6,
    )

    # retrieve the results for every qubit
    for pair in qubit_pairs:
        # WHEN MULTIPLEXING IS WORKING
        # for state, qubit in zip([0, 1], pair):
        for state, qubit in zip([1], [pair[1]]):
            # average msr, phase, i and q over the number of shots defined in the runcard
            ro_pulse = ro_pulses[qubit]
            result = results[ro_pulse.serial]

            prob = np.abs(
                result.i
                + 1j * result.q
                - complex(platform.qubits[qubit].mean_gnd_states)
            ) / np.abs(
                complex(platform.qubits[qubit].mean_exc_states)
                - complex(platform.qubits[qubit].mean_gnd_states)
            )
            # prob = iq_to_probability(result.i, result.q, complex(platform.qubits[qubit].mean_exc_states), complex(platform.qubits[qubit].mean_gnd_states))
            # store the results
            r = {
                "frequency[Hz]": delta_frequency_range + coupler_frequency,
                "coupler": len(delta_frequency_range) * [f"c{pair[0]}"],
                "qubit": len(delta_frequency_range) * [qubit],
                "state": len(delta_frequency_range) * [state],
                "probability": prob,
            }
            sweep_data.add_data_from_dict(r)

    # Temporary fix for multiplexing, repeat the experiment for the second qubit
    ro_pulses[pair[0]] = platform.create_MZ_pulse(
        pair[0], start=ro_pulses[pair[1]].se_finish
    )  # Multiplex not working yet
    sequence.add(ro_pulses[pair[0]])
    sequence.remove(ro_pulses[pair[1]])
    results = platform.sweep(
        sequence,
        sweeper,
        nshots=nshots,
        acquisition_type=AcquisitionType.INTEGRATION,  # AcquisitionType.DISCRIMINATION
        averaging_mode=AveragingMode.CYCLIC,  # AcquisitionMode.CYCLIC
        relaxation_time=300e-6,
    )
    for pair in qubit_pairs:
        # WHEN MULTIPLEXING IS WORKING
        # for state, qubit in zip([0, 1], pair):
        for state, qubit in zip([0], [pair[0]]):
            # average msr, phase, i and q over the number of shots defined in the runcard
            ro_pulse = ro_pulses[qubit]
            result = results[ro_pulse.serial]

            prob = np.abs(
                result.i
                + 1j * result.q
                - complex(platform.qubits[qubit].mean_gnd_states)
            ) / np.abs(
                complex(platform.qubits[qubit].mean_exc_states)
                - complex(platform.qubits[qubit].mean_gnd_states)
            )
            # store the results
            r = {
                "frequency[Hz]": delta_frequency_range + coupler_frequency,
                "coupler": len(delta_frequency_range) * [f"c{pair[0]}"],
                "qubit": len(delta_frequency_range) * [qubit],
                "state": len(delta_frequency_range) * [state],
                "probability": prob,
            }
            sweep_data.add_data_from_dict(r)

    yield sweep_data


@plot("coupler_swap", plots.coupler_swap_amplitude)
def coupler_swap_amplitude(
    platform: AbstractPlatform,
    qubits: dict,
    coupler_frequency: float,
    coupler_drive_duration: float,
    amplitude_min: float,
    amplitude_max: float,
    amplitude_step: float,
    nshots: int,
):
    r"""
    Perform a SWAP experiment between two qubits through the coupler by changing its amplitude.

    Args:
        platform (AbstractPlatform): The quantum platform to run the experiment on.
        qubits (dict): The qubits to perform the experiment on.
        coupler_frequency (float): The frequency of the coupler.
        coupler_drive_duration (float): The duration of the coupler drive pulse.
        amplitude_min (float): The minimum amplitude of the coupler drive pulse.
        amplitude_max (float): The maximum amplitude of the coupler drive pulse.
        amplitude_step (float): The step size of the amplitude of the coupler drive pulse.
        nshots (int): The number of shots to average over.

    Returns:
        - DataUnits: The results of the experiment.
            - **amplitude[dimensionless]**: The amplitude of the coupler.
            - **coupler** (str): The name of the coupler.
            - **qubit** (str): The name of the qubit measured.
            - **probability** (float): The probability of the qubit being in the excited state.
            - **state** (str): The state of the qubit.
    """
    # define the qubit to measure
    # FIXME: make general for multiple qubits
    if 2 in qubits:
        qubits.pop(2)
    qubit_pairs = list([qubit, 2] for qubit in qubits)

    # create a sequence
    sequence = PulseSequence()
    ro_pulses = {}
    cd_pulses = {}
    qd_pulses = {}

    for pair in qubit_pairs:
        qd_pulses[pair[1]] = platform.create_RX_pulse(pair[1], start=0)
        cd_pulses[pair[0]] = platform.create_qubit_drive_pulse(
            pair[0], start=qd_pulses[pair[1]].se_finish, duration=coupler_drive_duration
        )
        cd_pulses[pair[0]].frequency = coupler_frequency
        platform.qubits[pair[0]].drive_frequency = coupler_frequency
        platform.qubits[pair[0]].drive.local_oscillator.frequency = (
            coupler_frequency - 100_000_000
        )

        ro_pulses[pair[1]] = platform.create_MZ_pulse(
            pair[1], start=cd_pulses[pair[0]].se_finish
        )
        # ro_pulses[pair[0]] = platform.create_MZ_pulse(pair[0], start=ro_pulses[pair[1]].se_finish) # Multiplex not working yet

        sequence.add(qd_pulses[pair[1]])
        sequence.add(cd_pulses[pair[0]])
        sequence.add(ro_pulses[pair[1]])
        # sequence.add(ro_pulses[pair[0]])

    # define the parameter to sweep and its range:
    amplitude_range = np.arange(amplitude_min, amplitude_max, amplitude_step)

    sweeper = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        pulses=[cd_pulses[pair[0]] for pair in qubit_pairs],
    )

    # create a DataUnits object to store the results,
    sweep_data = DataUnits(
        name="data",
        quantities={"amplitude": "dimensionless"},
        options=["coupler", "qubit", "state", "probability"],
    )

    # repeat the experiment as many times as defined by nshots
    results = platform.sweep(
        sequence,
        sweeper,
        nshots=nshots,
        acquisition_type=AcquisitionType.INTEGRATION,  # AcquisitionType.DISCRIMINATION
        averaging_mode=AveragingMode.CYCLIC,  # AcquisitionMode.CYCLIC
        relaxation_time=300e-6,
    )

    # retrieve the results for every qubit
    for pair in qubit_pairs:
        # WHEN MULTIPLEXING IS WORKING
        # for state, qubit in zip([0, 1], pair):
        for state, qubit in zip([1], [pair[1]]):
            # average msr, phase, i and q over the number of shots defined in the runcard
            ro_pulse = ro_pulses[qubit]
            result = results[ro_pulse.serial]

            prob = np.abs(
                result.i
                + 1j * result.q
                - complex(platform.qubits[qubit].mean_gnd_states)
            ) / np.abs(
                complex(platform.qubits[qubit].mean_exc_states)
                - complex(platform.qubits[qubit].mean_gnd_states)
            )
            # prob = iq_to_probability(result.i, result.q, complex(platform.qubits[qubit].mean_exc_states), complex(platform.qubits[qubit].mean_gnd_states))
            # store the results
            r = {
                "amplitude[dimensionless]": amplitude_range,
                "coupler": len(amplitude_range) * [f"c{pair[0]}"],
                "qubit": len(amplitude_range) * [qubit],
                "state": len(amplitude_range) * [state],
                "probability": prob,
            }
            sweep_data.add_data_from_dict(r)

    # Temporary fix for multiplexing, repeat the experiment for the second qubit
    ro_pulses[pair[0]] = platform.create_MZ_pulse(
        pair[0], start=ro_pulses[pair[1]].se_finish
    )  # Multiplex not working yet
    sequence.add(ro_pulses[pair[0]])
    sequence.remove(ro_pulses[pair[1]])
    results = platform.sweep(
        sequence,
        sweeper,
        nshots=nshots,
        acquisition_type=AcquisitionType.INTEGRATION,  # AcquisitionType.DISCRIMINATION
        averaging_mode=AveragingMode.CYCLIC,  # AcquisitionMode.CYCLIC
        relaxation_time=300e-6,
    )
    for pair in qubit_pairs:
        # WHEN MULTIPLEXING IS WORKING
        # for state, qubit in zip([0, 1], pair):
        for state, qubit in zip([0], [pair[0]]):
            # average msr, phase, i and q over the number of shots defined in the runcard
            ro_pulse = ro_pulses[qubit]
            result = results[ro_pulse.serial]

            prob = np.abs(
                result.i
                + 1j * result.q
                - complex(platform.qubits[qubit].mean_gnd_states)
            ) / np.abs(
                complex(platform.qubits[qubit].mean_exc_states)
                - complex(platform.qubits[qubit].mean_gnd_states)
            )
            # store the results
            r = {
                "amplitude[dimensionless]": amplitude_range,
                "coupler": len(amplitude_range) * [f"c{pair[0]}"],
                "qubit": len(amplitude_range) * [qubit],
                "state": len(amplitude_range) * [state],
                "probability": prob,
            }
            sweep_data.add_data_from_dict(r)

    yield sweep_data
