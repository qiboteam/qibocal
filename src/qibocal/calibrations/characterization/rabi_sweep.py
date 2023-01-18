import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Sweeper

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import rabi_fit

import time
from qibolab.result import ExecutionResult
from types import SimpleNamespace
from qm.qua import *
from qualang_tools.loops import from_array


@plot("MSR vs Time", plots.time_msr_phase)
def rabi_pulse_length_sweep(
    platform: AbstractPlatform,
    qubits: list,
    pulse_duration_start,
    pulse_duration_end,
    pulse_duration_step,
    nshots=1024,
    software_averages=1,
    points=10,
):

    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse duration
    to find the drive pulse length that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
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
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(qubit, start=0, duration=pulse_duration_start)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=pulse_duration_start)
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        pulse_duration_start, pulse_duration_end, pulse_duration_step
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse duration time
    data = DataUnits(
        name="data", quantities={"time": "ns"}, options=["qubit", "iteration"]
    )

    start_time = time.time()
    # repeat the experiment as many times as defined by software_averages
    #count = 0
    for iteration in range(software_averages):
        # sweep the parameter
        #results = platform.sweep(sequence, sweeper, nshots=1000)

        with program() as experiment:
            n = declare(int)
            outputs = {
                pulse.serial: SimpleNamespace(
                    I=declare(fixed),
                    Q=declare(fixed),
                    I_st=declare_stream(),
                    Q_st=declare_stream(),
                    threshold=None,
                )
                for pulse in sequence.ro_pulses
            }
            qmsequence = []
            for pulse in sequence:
                qmpulse = SimpleNamespace(
                    pulse=pulse, target=platform.design.opx.register_pulse(platform.qubits[pulse.qubit], pulse), operation=pulse.serial
                )
                qmsequence.append(qmpulse)

            with for_(n, 0, n < nshots, n + 1):
                dur = declare(int)
                with for_(*from_array(dur, qd_pulse_duration_range // 4)):
                    align()
                    for qmpulse in qmsequence:
                        pulse = qmpulse.pulse
                        if pulse.type.name == "READOUT":
                            wait(dur)
                            output = outputs[pulse.serial]
                            measure(
                                qmpulse.operation,
                                qmpulse.target,
                                None,
                                dual_demod.full("cos", "out1", "sin", "out2", output.I),
                                dual_demod.full("minus_sin", "out1", "cos", "out2", output.Q),
                            )
                        else:
                            play(qmpulse.operation, qmpulse.target, duration=dur)

                    wait(platform.relaxation_time // 4)
                    # Save data to the stream processing
                    for output in outputs.values():
                        save(output.I, output.I_st)
                        save(output.Q, output.Q_st)

            with stream_processing():
                for serial, output in outputs.items():
                    output.I_st.buffer(len(qd_pulse_duration_range)).average().save(f"{serial}_I")
                    output.Q_st.buffer(len(qd_pulse_duration_range)).average().save(f"{serial}_Q")

            # save data as often as defined by points
            #if count % points == 0 and count > 0:
            #    # save data
            #    yield data
            #    # calculate and save fit
            #    yield rabi_fit(
            #        data,
            #        x="time[ns]",
            #        y="MSR[uV]",
            #        qubits=qubits,
            #        resonator_type=platform.resonator_type,
            #        labels=[
            #            "pi_pulse_duration",
            #            "pi_pulse_peak_voltage",
            #        ],
            #    )

        # execute the pulse sequence
        #results = platform.execute_pulse_sequence(sequence, nshots=nshots)
        machine = platform.design.opx.manager.open_qm(platform.design.opx.config)

        # for debugging only
        from qm import generate_qua_script

        with open("qua_script.txt", "w") as file:
            file.write(generate_qua_script(experiment, platform.design.opx.config))

        job = machine.execute(experiment)
        handles = job.result_handles
        handles.wait_for_all_values()
        results = {}
        for serial in outputs.keys():
            ires = handles.get(f"{serial}_I").fetch_all()
            qres = handles.get(f"{serial}_Q").fetch_all()
            if f"{serial}_shots" in handles:
                shots = handles.get(f"{serial}_shots").fetch_all().astype(int)
            else:
                shots = None
            results[serial] = ExecutionResult(ires, qres, shots)

        for qubit in qubits:
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[qubit].serial]
            r = {
                "MSR[V]": result.MSR,
                "i[V]": result.I,
                "q[V]": result.Q,
                "phase[rad]": result.phase,
                "time[ns]": qd_pulse_duration_range,
                "qubit": len(qd_pulse_duration_range) * [qubit],
                "iteration": iteration,
            }
            data.add_data_from_dict(r)
        #count += 1
    print("Total execution time:", time.time() - start_time)
    yield data
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
