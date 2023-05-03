import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.platforms.platform import AcquisitionType, AveragingMode
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import rabi_fit


@plot("MSR vs Time", plots.time_msr_phase)
def rabi_pulse_length(
    platform: AbstractPlatform,
    qubits: dict,
    pulse_duration_start,
    pulse_duration_end,
    pulse_duration_step,
    nshots=1024,
    relaxation_time=None,
    software_averages=1,
):
    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse duration
    to find the drive pulse length that creates a rotation of a desired angle.
    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        pulse_duration_start (int): Initial drive pulse duration for the Rabi experiment
        pulse_duration_end (int): Maximum drive pulse duration for the Rabi experiment
        pulse_duration_step (int): Scan range step for the drive pulse duration for the Rabi experiment
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points
    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys
            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **time[ns]**: Drive pulse duration in ns
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages
        - A DataUnits object with the fitted data obtained with the following keys
            - **pi_pulse_duration**: pi pulse duration
            - **pi_pulse_peak_voltage**: pi pulse's maximum voltage
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
            - **qubit**: The qubit being tested
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )
    sweeper = Sweeper(
        Parameter.duration,
        qd_pulse_duration_range,
        [qd_pulses[qubit] for qubit in qubits],
    )
    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse duration time
    data = DataUnits(
        name="data", quantities={"time": "ns"}, options=["qubit", "iteration"]
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        # sweep the parameter
        results = platform.sweep(
            sequence,
            sweeper,
            nshots=nshots,
            relaxation_time=relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
        for qubit in qubits:
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[qubit].serial]
            r = result.raw
            r.update(
                {
                    "time[ns]": qd_pulse_duration_range,
                    "qubit": len(qd_pulse_duration_range) * [qubit],
                    "iteration": len(qd_pulse_duration_range) * [iteration],
                }
            )
            data.add_data_from_dict(r)
            # save data
        yield data
        # calculate and save fit
        yield rabi_fit(
            data,
            x="time[ns]",
            y="MSR[uV]",
            qubits=qubits,
            resonator_type=platform.resonator_type,
            labels=[
                "pi_pulse_duration",
                "pi_pulse_peak_voltage",
            ],
        )


@plot("MSR vs Amplitude", plots.amplitude_msr_phase)
def rabi_pulse_amplitude(
    platform: AbstractPlatform,
    qubits: dict,
    pulse_amplitude_start,
    pulse_amplitude_end,
    pulse_amplitude_step,
    nshots=1024,
    relaxation_time=None,
    software_averages=1,
):
    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        pulse_amplitude_start (int): Initial drive pulse amplitude for the Rabi experiment
        pulse_amplitude_end (int): Maximum drive pulse amplitude for the Rabi experiment
        pulse_amplitude_step (int): Scan range step for the drive pulse amplitude for the Rabi experiment
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points
    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys
            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **amplitude[dimensionless]**: Drive pulse amplitude
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages
        - A DataUnits object with the fitted data obtained with the following keys
            - **pi_pulse_amplitude**: pi pulse amplitude
            - **pi_pulse_peak_voltage**: pi pulse's maximum voltage
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
            - **qubit**: The qubit being tested
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse amplitude
    qd_pulse_amplitude_range = np.arange(
        pulse_amplitude_start, pulse_amplitude_end, pulse_amplitude_step
    )
    sweeper = Sweeper(
        Parameter.amplitude,
        qd_pulse_amplitude_range,
        [qd_pulses[qubit] for qubit in qubits],
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse amplitude
    data = DataUnits(
        name=f"data",
        quantities={"amplitude": "dimensionless"},
        options=["qubit", "iteration"],
    )

    for iteration in range(software_averages):
        # sweep the parameter
        results = platform.sweep(
            sequence,
            sweeper,
            nshots=nshots,
            relaxation_time=relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
        for qubit in qubits:
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[qubit].serial]
            r = result.raw
            r.update(
                {
                    "amplitude[dimensionless]": qd_pulse_amplitude_range,
                    "qubit": len(qd_pulse_amplitude_range) * [qubit],
                    "iteration": len(qd_pulse_amplitude_range) * [iteration],
                }
            )
            data.add_data_from_dict(r)

        yield data
        # calculate and save fit
        yield rabi_fit(
            data,
            x="amplitude[dimensionless]",
            y="MSR[uV]",
            qubits=qubits,
            resonator_type=platform.resonator_type,
            labels=[
                "pi_pulse_amplitude",
                "pi_pulse_peak_voltage",
            ],
        )


# TODO: Is this actually useful ???
@plot("MSR vs length and amplitude", plots.duration_amplitude_msr_phase)
def rabi_pulse_length_and_amplitude(
    platform,
    qubits: dict,
    pulse_duration_start,
    pulse_duration_end,
    pulse_duration_step,
    pulse_amplitude_start,
    pulse_amplitude_end,
    pulse_amplitude_step,
    nshots=1024,
    relaxation_time=None,
    software_averages=1,
):
    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse
    combination of duration and amplitude to find the drive pulse amplitude that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        pulse_duration_start (int): Initial drive pulse duration for the Rabi experiment
        pulse_duration_end (int): Maximum drive pulse duration for the Rabi experiment
        pulse_duration_step (int): Scan range step for the drive pulse duration for the Rabi experiment
        pulse_amplitude_start (int): Initial drive pulse amplitude for the Rabi experiment
        pulse_amplitude_end (int): Maximum drive pulse amplitude for the Rabi experiment
        pulse_amplitude_step (int): Scan range step for the drive pulse amplitude for the Rabi experiment
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **duration[ns]**: Drive pulse duration in ns
            - **amplitude[dimensionless]**: Drive pulse amplitude
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )
    sweeper_duration = Sweeper(
        Parameter.duration,
        qd_pulse_duration_range,
        [qd_pulses[qubit] for qubit in qubits],
    )
    # qubit drive pulse amplitude
    qd_pulse_amplitude_range = np.arange(
        pulse_amplitude_start, pulse_amplitude_end, pulse_amplitude_step
    )
    sweeper_amplitude = Sweeper(
        Parameter.amplitude,
        qd_pulse_amplitude_range,
        [qd_pulses[qubit] for qubit in qubits],
    )

    # create a DataUnits object to store the results
    # that includes qubit drive pulse duration and amplitude
    data = DataUnits(
        name=f"data",
        quantities={"duration": "ns", "amplitude": "dimensionless"},
        options=["qubit", "iteration"],
    )

    for iteration in range(software_averages):
        results = platform.sweep(
            sequence,
            sweeper_duration,
            sweeper_amplitude,
            relaxation_time=relaxation_time,
            nshots=nshots,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
        for ro_pulse in ro_pulses.values():
            # average msr, phase, i and q over the number of shots defined in the runcard
            # r = results[ro_pulse.serial].average.raw
            result = results[ro_pulse.serial]
            # store the results
            durations = np.repeat(
                qd_pulse_amplitude_range, len(qd_pulse_duration_range)
            )
            amps = np.array(
                len(qd_pulse_duration_range) * list(qd_pulse_amplitude_range)
            ).flatten()
            r = {k: v.ravel() for k, v in result.raw.items()}
            r.update(
                {
                    "duration[ns]": durations,
                    "amplitude[dimensionless]": amps,
                    "qubit": len(durations) * ro_pulse.qubit,
                    "iteration": len(durations) * [iteration],
                }
            )
            data.add_data_from_dict(r)
    yield data
