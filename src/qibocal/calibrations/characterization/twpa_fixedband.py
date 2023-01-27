import time

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Sweeper

from qibocal import plots
from qibocal.data import Data, DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import calibrate_qubit_states_fit


@plot("TWPA frequency", plots.twpa_frequency)
def twpa_frequency(
    platform: AbstractPlatform,
    qubits: list,
    frequency_width: float,
    frequency_step: float,
    nshots,
):
    """
    Method which optimizes the Read-out fidelity by varying the Read-out pulse frequency.
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
    initial_frequency = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )
        initial_frequency[qubit] = platform.qubits[
            qubit
        ].twpa.local_oscillator.frequency

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
    ).astype(int)

    # retrieve and store the results for every qubit
    start_time = time.time()
    for frequency in delta_frequency_range:
        for qubit in qubits:
            platform.qubits[qubit].twpa.local_oscillator.frequency = (
                frequency + initial_frequency[qubit]
            )

        state0_results = platform.execute_pulse_sequence(state0_sequence, nshots=nshots)
        for qubit in qubits:
            r = state0_results[ro_pulses[qubit].serial].to_dict(average=False)
            r.update(
                {
                    "frequency[Hz]": [
                        platform.qubits[qubit].twpa.local_oscillator.frequency
                    ]
                    * nshots,
                    "delta_frequency[Hz]": [frequency] * nshots,
                    "qubit": [qubit] * nshots,
                    "iteration": np.arange(
                        nshots
                    ),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                    "state": [0] * nshots,
                }
            )
            data.add_data_from_dict(r)
        print("State0 saving time:", time.time() - start_time)
        yield data
    print("State0 run time:", time.time() - start_time)

    # retrieve and store the results for every qubit
    start_time = time.time()
    for frequency in delta_frequency_range:
        for qubit in qubits:
            platform.qubits[qubit].twpa.local_oscillator.frequency = (
                frequency + initial_frequency[qubit]
            )

        state1_results = platform.execute_pulse_sequence(state1_sequence, nshots=nshots)
        for qubit in qubits:
            r = state1_results[ro_pulses[qubit].serial].to_dict(average=False)
            r.update(
                {
                    "frequency[Hz]": [
                        platform.qubits[qubit].twpa.local_oscillator.frequency
                    ]
                    * nshots,
                    "delta_frequency[Hz]": [frequency] * nshots,
                    "qubit": [qubit] * nshots,
                    "iteration": np.arange(
                        nshots
                    ),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                    "state": [1] * nshots,
                }
            )
            data.add_data_from_dict(r)
        print("State1 saving time:", time.time() - start_time)

        # finally, save the remaining data and the fits
        yield data

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


@plot("TWPA power", plots.twpa_power)
def twpa_power(
    platform: AbstractPlatform,
    qubits: list,
    power_width: float,
    power_step: float,
    nshots,
):
    """
    Method which optimizes the Read-out fidelity by varying the Read-out pulse power.
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
    initial_power = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )
        initial_power[qubit] = platform.qubits[qubit].twpa.local_oscillator.power

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    # create a DataUnits object to store the results
    data = DataUnits(
        name="data",
        quantities={"power": "dBm", "delta_power": "dBm"},
        options=["qubit", "iteration", "state"],
    )
    data_fit = Data(
        name="fit",
        quantities=[
            "power",
            "delta_power",
            "rotation_angle",
            "threshold",
            "fidelity",
            "assignment_fidelity",
            "average_state0",
            "average_state1",
            "qubit",
        ],
    )

    # iterate over the power range
    delta_power_range = np.arange(-power_width / 2, power_width / 2, power_step).astype(
        int
    )

    # retrieve and store the results for every qubit
    start_time = time.time()
    for power in delta_power_range:
        for qubit in qubits:
            platform.qubits[qubit].twpa.local_oscillator.power = (
                power + initial_power[qubit]
            )

        state0_results = platform.execute_pulse_sequence(state0_sequence, nshots=nshots)
        for qubit in qubits:
            r = state0_results[ro_pulses[qubit].serial].to_dict(average=False)
            r.update(
                {
                    "power[dBm]": [platform.qubits[qubit].twpa.local_oscillator.power]
                    * nshots,
                    "delta_power[dBm]": [power] * nshots,
                    "qubit": [qubit] * nshots,
                    "iteration": np.arange(
                        nshots
                    ),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                    "state": [0] * nshots,
                }
            )
            data.add_data_from_dict(r)
        print("State0 saving time:", time.time() - start_time)
        yield data
    print("State0 run time:", time.time() - start_time)

    # retrieve and store the results for every qubit
    start_time = time.time()
    for power in delta_power_range:
        for qubit in qubits:
            platform.qubits[qubit].twpa.local_oscillator.power = (
                power + initial_power[qubit]
            )

        state1_results = platform.execute_pulse_sequence(state1_sequence, nshots=nshots)
        for qubit in qubits:
            r = state1_results[ro_pulses[qubit].serial].to_dict(average=False)
            r.update(
                {
                    "power[dBm]": [platform.qubits[qubit].twpa.local_oscillator.power]
                    * nshots,
                    "delta_power[dBm]": [power] * nshots,
                    "qubit": [qubit] * nshots,
                    "iteration": np.arange(
                        nshots
                    ),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                    "state": [1] * nshots,
                }
            )
            data.add_data_from_dict(r)
        print("State1 saving time:", time.time() - start_time)

        # finally, save the remaining data and the fits
        yield data

    # fit the data
    for delta_freq in delta_power_range:
        import copy

        import pandas as pd

        start_time = time.time()
        data_trim = copy.deepcopy(data)
        data_trim.df = data_trim.df[
            data_trim.get_values("delta_power", "dBm") == delta_freq
        ]

        fits = calibrate_qubit_states_fit(
            data_trim, x="i[V]", y="q[V]", nshots=nshots, qubits=qubits
        )
        fits.df["delta_power"] = [delta_freq] * len(qubits)
        data_fit.df = pd.concat([data_fit.df, fits.df], ignore_index=True)
        print("Fitting time:", time.time() - start_time)
        yield data_fit
