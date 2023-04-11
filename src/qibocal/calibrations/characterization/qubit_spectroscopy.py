from typing import Optional

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Qubit Drive Frequency", plots.frequency_msr_phase)
def qubit_spectroscopy(
    platform: AbstractPlatform,
    qubits: dict,
    freq_width: int,
    freq_step: int,
    drive_duration: int,
    drive_amplitude: Optional[float] = None,
    nshots: int = 1024,
    relaxation_time: int = 50,
    software_averages: int = 1,
):
    r"""
    Perform spectroscopy on the qubit.
    This routine executes a fast scan around the expected qubit frequency indicated in the platform runcard.
    Afterthat, a final sweep with more precision is executed centered in the new qubit frequency found.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        freq_width (int): Width frequency in HZ to perform the high resolution sweep
        freq_step (int): Step frequency in HZ for the high resolution sweep
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
            qubit, start=0, duration=drive_duration
        )
        if drive_amplitude is not None:
            qd_pulses[qubit].amplitude = drive_amplitude
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit frequency
    data = DataUnits(
        name="data",
        quantities={"frequency": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        results = platform.sweep(
            sequence, sweeper, nshots=nshots, relaxation_time=relaxation_time
        )

        # retrieve the results for every qubit
        for qubit, ro_pulse in ro_pulses.items():
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulse.serial]
            r = result.raw
            # store the results
            r.update(
                {
                    "frequency[Hz]": delta_frequency_range + qd_pulses[qubit].frequency,
                    "qubit": len(delta_frequency_range) * [qubit],
                    "iteration": len(delta_frequency_range) * [iteration],
                }
            )
            data.add_data_from_dict(r)

        # finally, save the remaining data and fits
        yield data
        yield lorentzian_fit(
            data,
            x="frequency[GHz]",
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
    drive_amplitude,
    freq_width,
    freq_step,
    bias_width,
    bias_step,
    fluxlines,
    nshots=1024,
    relaxation_time=50,
    software_averages=1,
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
            qubit, start=0, duration=2000
        )
        qd_pulses[qubit].amplitude = drive_amplitude
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive frequency
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
    frequency_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
    )

    if fluxlines == "qubits":
        fluxlines = qubits

    # flux bias
    delta_bias_range = np.arange(-bias_width / 2, bias_width / 2, bias_step)
    bias_sweeper = Sweeper(
        Parameter.bias,
        delta_bias_range,
        qubits=fluxlines,
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
        results = platform.sweep(
            sequence,
            bias_sweeper,
            frequency_sweeper,
            nshots=nshots,
            relaxation_time=relaxation_time,
        )

        # retrieve the results for every qubit
        for qubit, fluxline in zip(qubits, fluxlines):
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[qubit].serial]
            # store the results
            biases = np.repeat(
                delta_bias_range, len(delta_frequency_range)
            ) + platform.get_bias(fluxline)
            freqs = np.array(
                len(delta_bias_range)
                * list(delta_frequency_range + qd_pulses[qubit].frequency)
            ).flatten()
            r = {k: v.ravel() for k, v in result.raw.items()}
            r.update(
                {
                    "frequency[Hz]": freqs,
                    "bias[V]": biases,
                    "qubit": len(freqs) * [qubit],
                    "fluxline": len(freqs) * [fluxline],
                    "iteration": len(freqs) * [iteration],
                }
            )
            data.add_data_from_dict(r)

        # finally, save the remaining data and fits
        yield data
