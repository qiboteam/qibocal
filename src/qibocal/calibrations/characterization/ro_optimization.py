import time

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Sweeper

from qibocal import plots
from qibocal.data import Data, DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import calibrate_qubit_states_fit


@plot("Qubit States", plots.ro_frequency)
def ro_frequency(
    platform: AbstractPlatform,
    qubits: list,
    frequency_width: float,
    frequency_step: float,
    nshots,
):
    """
    Method which optimizes the Read-out fidelity by varying the Read-out pulse duration.
    Two analogous tests are performed for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.
    Their distinctiveness is then associated to the fidelity.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (list): List of target qubits to perform the action
        nshots (int): number of times the pulse sequence will be repeated.
    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **iteration[ns]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    # create a DataUnits object to store the results
    data = DataUnits(
        name="data",
        quantities={"frequency": "Hz", "delta_frequency": "Hz"},
        options=["qubit", "iteration", "state"],
    )
    data_fit = Data(
        name="fit",
        quantities=[
            "frequency",
            "delta_frequency",
            "rotation_angle",
            "threshold",
            "fidelity",
            "assignment_fidelity",
            "average_state0",
            "average_state1",
            "qubit",
        ],
    )

    # iterate over the frequency range
    delta_frequency_range = np.arange(
        -frequency_width / 2, frequency_width / 2, frequency_step
    )

    frequency_sweeper = Sweeper(
        "frequency", delta_frequency_range, [ro_pulses[qubit] for qubit in qubits]
    )

    # execute the first pulse sequence
    start_time = time.time()
    state0_results = platform.sweep(
        state0_sequence, frequency_sweeper, nshots=nshots, average=False
    )
    print("State0 run time:", time.time() - start_time)

    # retrieve and store the results for every qubit
    start_time = time.time()
    while any(result.in_progress for result in state0_results.values()) or True:
        for qubit in qubits:
            result = state0_results[ro_pulses[qubit].serial]
            r = {
                "MSR[V]": result.MSR.flatten(),
                "i[V]": result.I.flatten(),
                "q[V]": result.Q.flatten(),
                "phase[rad]": result.phase.flatten(),
                "frequency[Hz]": [ro_pulses[qubit].frequency]
                * nshots
                * len(delta_frequency_range),
                "delta_frequency[Hz]": np.repeat(
                    np.vstack(delta_frequency_range).T, len(np.arange(nshots)), axis=0
                ).flatten(),
                "qubit": [qubit] * nshots * len(delta_frequency_range),
                "iteration": np.repeat(
                    np.vstack(np.arange(nshots)).T, len(delta_frequency_range), axis=1
                ).flatten(),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [0] * nshots * len(delta_frequency_range),
            }
            data.add_data_from_dict(r)
        print("State0 saving time:", time.time() - start_time)
        yield data
        # FIXME: Remove the While True and break once the result.in_progress works
        break

    # execute the second pulse sequence
    start_time = time.time()
    state1_results = platform.sweep(
        state1_sequence, frequency_sweeper, nshots=nshots, average=False
    )
    print("State1 time:", time.time() - start_time)

    # retrieve and store the results for every qubit
    start_time = time.time()
    while any(result.in_progress for result in state1_results.values()) or True:
        for qubit in qubits:
            result = state1_results[ro_pulses[qubit].serial]
            r = {
                "MSR[V]": result.MSR.flatten(),
                "i[V]": result.I.flatten(),
                "q[V]": result.Q.flatten(),
                "phase[rad]": result.phase.flatten(),
                "frequency[Hz]": [ro_pulses[qubit].frequency]
                * nshots
                * len(delta_frequency_range),
                "delta_frequency[Hz]": np.repeat(
                    np.vstack(delta_frequency_range).T, len(np.arange(nshots)), axis=0
                ).flatten(),
                "qubit": [qubit] * nshots * len(delta_frequency_range),
                "iteration": np.repeat(
                    np.vstack(np.arange(nshots)).T, len(delta_frequency_range), axis=1
                ).flatten(),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [1] * nshots * len(delta_frequency_range),
            }
            data.add_data_from_dict(r)
        print("State1 saving time:", time.time() - start_time)

        # finally, save the remaining data and the fits
        yield data

        # FIXME: Remove the While True and break once the result.in_progress works
        break

    # fit the data
    for delta_freq in delta_frequency_range:
        import copy

        import pandas as pd

        start_time = time.time()
        data_trim = copy.deepcopy(data)
        data_trim.df = data_trim.df[
            data_trim.get_values("delta_frequency", "Hz") == delta_freq
        ]

        fits = calibrate_qubit_states_fit(
            data_trim, x="i[V]", y="q[V]", nshots=nshots, qubits=qubits
        )
        fits.df["delta_frequency"] = [delta_freq] * len(qubits)
        data_fit.df = pd.concat([data_fit.df, fits.df], ignore_index=True)
        print("Fitting time:", time.time() - start_time)
        yield data_fit


@plot("Qubit States", plots.ro_amplitude)
def ro_amplitude(
    platform: AbstractPlatform,
    qubits: list,
    amplitude_width: float,
    amplitude_step: float,
    nshots,
):
    """
    Method which optimizes the Read-out fidelity by varying the Read-out pulse amplitude.
    Two analogous tests are performed for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.
    Their distinctiveness is then associated to the fidelity.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (list): List of target qubits to perform the action
        nshots (int): number of times the pulse sequence will be repeated.
    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **iteration[dimensionless]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    # create a DataUnits object to store the results
    data = DataUnits(
        name="data",
        quantities={"amplitude": "dimensionless", "delta_amplitude": "dimensionless"},
        options=["qubit", "iteration", "state"],
    )
    data_fit = Data(
        name="fit",
        quantities=[
            "amplitude",
            "delta_amplitude",
            "rotation_angle",
            "threshold",
            "fidelity",
            "assignment_fidelity",
            "average_state0",
            "average_state1",
            "qubit",
        ],
    )

    # iterate over the amplitude range
    delta_amplitude_range = np.arange(
        1 - amplitude_width / 2, 1 + amplitude_width / 2, amplitude_step
    )

    amplitude_sweeper = Sweeper(
        "amplitude", delta_amplitude_range, [ro_pulses[qubit] for qubit in qubits]
    )

    # execute the first pulse sequence
    start_time = time.time()
    state0_results = platform.sweep(
        state0_sequence, amplitude_sweeper, nshots=nshots, average=False
    )
    print("State0 run time:", time.time() - start_time)

    # retrieve and store the results for every qubit
    start_time = time.time()
    while any(result.in_progress for result in state0_results.values()) or True:
        for qubit in qubits:
            result = state0_results[ro_pulses[qubit].serial]
            r = {
                "MSR[V]": result.MSR.flatten(),
                "i[V]": result.I.flatten(),
                "q[V]": result.Q.flatten(),
                "phase[rad]": result.phase.flatten(),
                "amplitude[dimensionless]": [ro_pulses[qubit].amplitude]
                * nshots
                * len(delta_amplitude_range),
                "delta_amplitude[dimensionless]": np.repeat(
                    np.vstack(delta_amplitude_range).T, len(np.arange(nshots)), axis=0
                ).flatten(),
                "qubit": [qubit] * nshots * len(delta_amplitude_range),
                "iteration": np.repeat(
                    np.vstack(np.arange(nshots)).T, len(delta_amplitude_range), axis=1
                ).flatten(),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [0] * nshots * len(delta_amplitude_range),
            }
            data.add_data_from_dict(r)
        print("State0 saving time:", time.time() - start_time)
        yield data
        # FIXME: Remove the While True and break once the result.in_progress works
        break

    # execute the second pulse sequence
    start_time = time.time()
    state1_results = platform.sweep(
        state1_sequence, amplitude_sweeper, nshots=nshots, average=False
    )
    print("State1 time:", time.time() - start_time)

    # retrieve and store the results for every qubit
    start_time = time.time()
    while any(result.in_progress for result in state1_results.values()) or True:
        for qubit in qubits:
            result = state1_results[ro_pulses[qubit].serial]
            r = {
                "MSR[V]": result.MSR.flatten(),
                "i[V]": result.I.flatten(),
                "q[V]": result.Q.flatten(),
                "phase[rad]": result.phase.flatten(),
                "amplitude[dimensionless]": [ro_pulses[qubit].amplitude]
                * nshots
                * len(delta_amplitude_range),
                "delta_amplitude[dimensionless]": np.repeat(
                    np.vstack(delta_amplitude_range).T, len(np.arange(nshots)), axis=0
                ).flatten(),
                "qubit": [qubit] * nshots * len(delta_amplitude_range),
                "iteration": np.repeat(
                    np.vstack(np.arange(nshots)).T, len(delta_amplitude_range), axis=1
                ).flatten(),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [1] * nshots * len(delta_amplitude_range),
            }
            data.add_data_from_dict(r)
        print("State1 saving time:", time.time() - start_time)

        # finally, save the remaining data and the fits
        yield data

        # FIXME: Remove the While True and break once the result.in_progress works
        break

    # fit the data
    for delta_freq in delta_amplitude_range:
        import copy

        import pandas as pd

        start_time = time.time()
        data_trim = copy.deepcopy(data)
        data_trim.df = data_trim.df[
            data_trim.get_values("delta_amplitude", "dimensionless") == delta_freq
        ]

        fits = calibrate_qubit_states_fit(
            data_trim, x="i[V]", y="q[V]", nshots=nshots, qubits=qubits
        )
        fits.df["delta_amplitude"] = [delta_freq] * len(qubits)
        data_fit.df = pd.concat([data_fit.df, fits.df], ignore_index=True)
        print("Fitting time:", time.time() - start_time)
        yield data_fit


@plot("Qubit States", plots.ro_duration)
def ro_duration_sweep(
    platform: AbstractPlatform,
    qubits: list,
    duration_width: float,
    duration_step: float,
    nshots,
):
    """
    Method which optimizes the Read-out fidelity by varying the Read-out pulse duration.
    Two analogous tests are performed for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.
    Their distinctiveness is then associated to the fidelity.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (list): List of target qubits to perform the action
        nshots (int): number of times the pulse sequence will be repeated.
    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **iteration[dimensionless]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    # create a DataUnits object to store the results
    data = DataUnits(
        name="data",
        quantities={"duration": "ns", "delta_duration": "ns"},
        options=["qubit", "iteration", "state"],
    )
    data_fit = Data(
        name="fit",
        quantities=[
            "duration",
            "delta_duration",
            "rotation_angle",
            "threshold",
            "fidelity",
            "assignment_fidelity",
            "average_state0",
            "average_state1",
            "qubit",
        ],
    )

    # iterate over the duration range
    delta_duration_range = np.arange(
        list(ro_pulses.values())[-1].duration - duration_width / 2,
        list(ro_pulses.values())[-1].duration + duration_width / 2,
        duration_step,
    )

    duration_sweeper = Sweeper(
        "duration", delta_duration_range, [ro_pulses[qubit] for qubit in qubits]
    )

    # execute the first pulse sequence
    start_time = time.time()
    state0_results = platform.sweep(
        state0_sequence, duration_sweeper, nshots=nshots, average=False
    )
    print("State0 run time:", time.time() - start_time)

    # retrieve and store the results for every qubit
    start_time = time.time()
    while any(result.in_progress for result in state0_results.values()) or True:
        for qubit in qubits:
            result = state0_results[ro_pulses[qubit].serial]
            r = {
                "MSR[V]": result.MSR.flatten(),
                "i[V]": result.I.flatten(),
                "q[V]": result.Q.flatten(),
                "phase[rad]": result.phase.flatten(),
                "duration[ns]": [ro_pulses[qubit].duration]
                * nshots
                * len(delta_duration_range),
                "delta_duration[ns]": np.repeat(
                    np.vstack(delta_duration_range).T, len(np.arange(nshots)), axis=0
                ).flatten(),
                "qubit": [qubit] * nshots * len(delta_duration_range),
                "iteration": np.repeat(
                    np.vstack(np.arange(nshots)).T, len(delta_duration_range), axis=1
                ).flatten(),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [0] * nshots * len(delta_duration_range),
            }
            data.add_data_from_dict(r)
        print("State0 saving time:", time.time() - start_time)
        yield data
        # FIXME: Remove the While True and break once the result.in_progress works
        break

    # execute the second pulse sequence
    start_time = time.time()
    state1_results = platform.sweep(
        state1_sequence, duration_sweeper, nshots=nshots, average=False
    )
    print("State1 time:", time.time() - start_time)

    # retrieve and store the results for every qubit
    start_time = time.time()
    while any(result.in_progress for result in state1_results.values()) or True:
        for qubit in qubits:
            result = state1_results[ro_pulses[qubit].serial]
            r = {
                "MSR[V]": result.MSR.flatten(),
                "i[V]": result.I.flatten(),
                "q[V]": result.Q.flatten(),
                "phase[rad]": result.phase.flatten(),
                "duration[ns]": [ro_pulses[qubit].duration]
                * nshots
                * len(delta_duration_range),
                "delta_duration[ns]": np.repeat(
                    np.vstack(delta_duration_range).T, len(np.arange(nshots)), axis=0
                ).flatten(),
                "qubit": [qubit] * nshots * len(delta_duration_range),
                "iteration": np.repeat(
                    np.vstack(np.arange(nshots)).T, len(delta_duration_range), axis=1
                ).flatten(),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [1] * nshots * len(delta_duration_range),
            }
            data.add_data_from_dict(r)
        print("State1 saving time:", time.time() - start_time)

        # finally, save the remaining data and the fits
        yield data

        # FIXME: Remove the While True and break once the result.in_progress works
        break

    # fit the data
    for delta_freq in delta_duration_range:
        import copy

        import pandas as pd

        start_time = time.time()
        data_trim = copy.deepcopy(data)
        data_trim.df = data_trim.df[
            data_trim.get_values("delta_duration", "ns") == delta_freq
        ]

        fits = calibrate_qubit_states_fit(
            data_trim, x="i[V]", y="q[V]", nshots=nshots, qubits=qubits
        )
        fits.df["delta_duration"] = [delta_freq] * len(qubits)
        data_fit.df = pd.concat([data_fit.df, fits.df], ignore_index=True)
        print("Fitting time:", time.time() - start_time)
        yield data_fit


@plot("Qubit States", plots.ro_duration)
def ro_duration(
    platform: AbstractPlatform,
    qubits: list,
    duration_width: float,
    duration_step: float,
    nshots,
):
    """
    Method which optimizes the Read-out fidelity by varying the Read-out pulse duration.
    Two analogous tests are performed for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.
    Their distinctiveness is then associated to the fidelity.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (list): List of target qubits to perform the action
        nshots (int): number of times the pulse sequence will be repeated.
    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **iteration[dimensionless]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    # create a DataUnits object to store the results
    data = DataUnits(
        name="data",
        quantities={"duration": "ns", "delta_duration": "ns"},
        options=["qubit", "iteration", "state"],
    )
    data_fit = Data(
        name="fit",
        quantities=[
            "duration",
            "delta_duration",
            "rotation_angle",
            "threshold",
            "fidelity",
            "assignment_fidelity",
            "average_state0",
            "average_state1",
            "qubit",
        ],
    )

    # iterate over the duration range
    delta_duration_range = (
        np.arange(-duration_width / 2, duration_width / 2, duration_step).astype(int)
        + ro_pulses[qubit].duration
    )

    duration_sweeper = Sweeper(
        "duration", delta_duration_range, [ro_pulses[qubit] for qubit in qubits]
    )

    # retrieve and store the results for every qubit
    start_time = time.time()
    for duration in delta_duration_range:
        for qubit in qubits:
            ro_pulses[qubit].duration = duration

        state0_results = platform.execute_pulse_sequence(state0_sequence, nshots=nshots)
        for qubit in qubits:
            result = state0_results[ro_pulses[qubit].serial]
            r = {
                "MSR[V]": result.MSR,
                "i[V]": result.I,
                "q[V]": result.Q,
                "phase[rad]": result.phase,
                "duration[ns]": [ro_pulses[qubit].duration] * nshots,
                "delta_duration[ns]": [duration] * nshots,
                "qubit": [qubit] * nshots,
                "iteration": np.arange(
                    nshots
                ),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [0] * nshots,
            }
            data.add_data_from_dict(r)
        print("State0 saving time:", time.time() - start_time)
        yield data
    print("State0 run time:", time.time() - start_time)

    # retrieve and store the results for every qubit
    start_time = time.time()
    for duration in delta_duration_range:
        for qubit in qubits:
            ro_pulses[qubit].duration = duration

        state1_results = platform.execute_pulse_sequence(state1_sequence, nshots=nshots)
        for qubit in qubits:
            result = state1_results[ro_pulses[qubit].serial]
            r = {
                "MSR[V]": result.MSR,
                "i[V]": result.I,
                "q[V]": result.Q,
                "phase[rad]": result.phase,
                "duration[ns]": [ro_pulses[qubit].duration] * nshots,
                "delta_duration[ns]": [duration] * nshots,
                "qubit": [qubit] * nshots,
                "iteration": np.arange(
                    nshots
                ),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [1] * nshots,
            }
            data.add_data_from_dict(r)
        print("State1 saving time:", time.time() - start_time)

        # finally, save the remaining data and the fits
        yield data

    # fit the data
    for delta_freq in delta_duration_range:
        import copy

        import pandas as pd

        start_time = time.time()
        data_trim = copy.deepcopy(data)
        data_trim.df = data_trim.df[
            data_trim.get_values("delta_duration", "ns") == delta_freq
        ]

        fits = calibrate_qubit_states_fit(
            data_trim, x="i[V]", y="q[V]", nshots=nshots, qubits=qubits
        )
        fits.df["delta_duration"] = [delta_freq] * len(qubits)
        data_fit.df = pd.concat([data_fit.df, fits.df], ignore_index=True)
        print("Fitting time:", time.time() - start_time)
        yield data_fit
