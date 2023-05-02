import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal import plots
from qibocal.data import Data, DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import ro_optimization_fit


@plot("Qubit States", plots.ro_frequency)
def ro_frequency(
    platform: AbstractPlatform,
    qubits: dict,
    frequency_width: float,
    frequency_step: float,
    nshots: int,
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
        frequency_width (float): width of the frequency range to be swept in Hz.
        frequency_step (float): step of the frequency range to be swept in Hz.
    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **iteration**: Execution number
            - **qubit**: The qubit being tested
            - **state**: The state of the qubit being tested
            - **frequency[Hz]**: The frequency of the readout being tested
            - **delta_frequency[Hz]**: The frequency offset from the runcard value

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
    sequences = {0: state0_sequence, 1: state1_sequence}
    # create a DataUnits object to store the results
    data = DataUnits(
        name="data",
        quantities={"frequency": "Hz", "delta_frequency": "Hz"},
        options=["qubit", "iteration", "state"],
    )

    # iterate over the frequency range
    delta_frequency_range = np.arange(
        -frequency_width / 2, frequency_width / 2, frequency_step
    )

    frequency_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
    )

    # Execute sequences for both states
    for state in [0, 1]:
        results = platform.sweep(
            sequences[state], frequency_sweeper, nshots=nshots, average=False
        )

        # retrieve and store the results for every qubit)
        for qubit in qubits:
            r = {k: v.ravel() for k, v in results[ro_pulses[qubit].serial].raw.items()}
            r.update(
                {
                    "frequency[Hz]": np.repeat(
                        np.vstack(delta_frequency_range).T,
                        nshots,
                        axis=0,
                    ).flatten()
                    + ro_pulses[qubit].frequency,
                    "delta_frequency[Hz]": np.repeat(
                        np.vstack(delta_frequency_range).T,
                        nshots,
                        axis=0,
                    ).flatten(),
                    "qubit": [qubit] * nshots * len(delta_frequency_range),
                    "iteration": np.repeat(
                        np.vstack(np.arange(nshots)).T,
                        len(delta_frequency_range),
                        axis=1,
                    ).flatten(),
                    "state": [state] * nshots * len(delta_frequency_range),
                }
            )
            data.add_data_from_dict(r)

    # finally, save the remaining data and the fits
    yield data
    yield ro_optimization_fit(data, "state", "qubit", "iteration", "delta_frequency")


@plot("Qubit States", plots.ro_amplitude)
def ro_amplitude(
    platform: AbstractPlatform,
    qubits: dict,
    amplitude_factor_min: float,
    amplitude_factor_max: float,
    amplitude_factor_step: float,
    nshots: int,
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
        amplitude_factor_min (float): minimum amplitude factor to be swept.
        amplitude_factor_max (float): maximum amplitude factor to be swept.
        amplitude_factor_step (float): step of the amplitude factor to be swept.
    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **qubit**: The qubit being tested
            - **iteration**: Execution number
            - **state**: The state of the qubit being tested
            - **amplitude_factor**: The amplitude factor of the readout being tested
            - **delta_amplitude_factor**: The amplitude offset from the runcard value

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
    sequences = {0: state0_sequence, 1: state1_sequence}
    # create a DataUnits object to store the results
    data = DataUnits(
        name="data",
        quantities={"amplitude": "dimensionless", "delta_amplitude": "dimensionless"},
        options=["qubit", "iteration", "state"],
    )

    # iterate over the amplitude range
    delta_amplitude_range = np.arange(
        amplitude_factor_min, amplitude_factor_max, amplitude_factor_step
    )

    amplitude_sweeper = Sweeper(
        Parameter.amplitude,
        delta_amplitude_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
    )

    # Execute sequences for both states
    for state in [0, 1]:
        results = platform.sweep(
            sequences[state], amplitude_sweeper, nshots=nshots, average=False
        )

        # retrieve and store the results for every qubit)
        for qubit in qubits:
            r = {k: v.ravel() for k, v in results[ro_pulses[qubit].serial].raw.items()}
            r.update(
                {
                    "amplitude[dimensionless]": np.repeat(
                        np.vstack(delta_amplitude_range).T,
                        nshots,
                        axis=0,
                    ).flatten()
                    * ro_pulses[qubit].amplitude,
                    "delta_amplitude[dimensionless]": np.repeat(
                        np.vstack(delta_amplitude_range).T,
                        nshots,
                        axis=0,
                    ).flatten(),
                    "qubit": [qubit] * nshots * len(delta_amplitude_range),
                    "iteration": np.repeat(
                        np.vstack(np.arange(nshots)).T,
                        len(delta_amplitude_range),
                        axis=1,
                    ).flatten(),
                    "state": [state] * nshots * len(delta_amplitude_range),
                }
            )
            data.add_data_from_dict(r)

    # finally, save the remaining data and the fits
    yield data
    yield ro_optimization_fit(data, "state", "qubit", "iteration", "delta_amplitude")


@plot("TWPA frequency", plots.ro_frequency)
def twpa_frequency(
    platform: AbstractPlatform,
    qubits: dict,
    frequency_width: float,
    frequency_step: float,
    nshots: int,
):
    """
    Method which optimizes the Read-out fidelity by varying the frequency of the TWPA.
    Two analogous tests are performed for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.
    Their distinctiveness is then associated to the fidelity.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (list): List of target qubits to perform the action
        frequency_width (float): Frequency range to sweep in Hz
        frequency_step (float): Frequency step to sweep in Hz
        nshots (int): number of times the pulse sequence will be repeated.
    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **qubit**: The qubit being tested
            - **iteration**: Execution number
            - **state**: The state of the qubit being tested
            - **frequency**: The frequency of the TWPA being tested
            - **delta_frequency**: The frequency offset from the runcard value

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
        initial_frequency[qubit] = platform.get_lo_twpa_frequency(qubit)

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])
    sequences = {0: state0_sequence, 1: state1_sequence}
    # create a DataUnits object to store the results
    data = DataUnits(
        name="data",
        quantities={"frequency": "Hz", "delta_frequency": "Hz"},
        options=["qubit", "iteration", "state"],
    )

    # iterate over the frequency range
    delta_frequency_range = np.arange(
        -frequency_width / 2, frequency_width / 2, frequency_step
    ).astype(int)

    # retrieve and store the results for every qubit
    for frequency in delta_frequency_range:
        for qubit in qubits:
            platform.set_lo_twpa_frequency(qubit, initial_frequency[qubit] + frequency)

        # Execute the sequences for both states
        for state in [0, 1]:
            results = platform.execute_pulse_sequence(sequences[state], nshots=nshots)
            for qubit in qubits:
                r = results[ro_pulses[qubit].serial].raw
                r.update(
                    {
                        "frequency[Hz]": [platform.get_lo_twpa_frequency(qubit)]
                        * nshots,
                        "delta_frequency[Hz]": [frequency] * nshots,
                        "qubit": [qubit] * nshots,
                        "iteration": np.arange(nshots),
                        "state": [state] * nshots,
                    }
                )
                data.add_data_from_dict(r)

        # finally, save the remaining data and the fits
        yield data
        yield ro_optimization_fit(
            data, "delta_frequency", "state", "qubit", "iteration"
        )


@plot("TWPA power", plots.ro_power)
def twpa_power(
    platform: AbstractPlatform,
    qubits: dict,
    power_width: float,
    power_step: float,
    nshots: int,
):
    """
    Method which optimizes the Read-out fidelity by varying the power of the TWPA.
    Two analogous tests are performed for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.
    Their distinctiveness is then associated to the fidelity.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (list): List of target qubits to perform the action
        power_width (float): width of the power range to be scanned in dBm
        power_step (float): step of the power range to be scanned in dBm
        nshots (int): number of times the pulse sequence will be repeated.
    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **qubit**: The qubit being tested
            - **iteration**: Execution number
            - **state**: The state of the qubit being tested
            - **power**: The power of the TWPA being tested
            - **delta_power**: The power offset from the runcard value

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
        initial_power[qubit] = platform.get_lo_twpa_power(qubit)

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])
    sequences = {0: state0_sequence, 1: state1_sequence}
    # create a DataUnits object to store the results
    data = DataUnits(
        name="data",
        quantities={"power": "dBm", "delta_power": "dBm"},
        options=["qubit", "iteration", "state"],
    )

    # iterate over the power range
    delta_power_range = np.arange(-power_width / 2, power_width / 2, power_step)

    # retrieve and store the results for every qubit
    for power in delta_power_range:
        for qubit in qubits:
            platform.set_lo_twpa_power(qubit, initial_power[qubit] + power)

        # Execute the sequences for both states
        for state in [0, 1]:
            results = platform.execute_pulse_sequence(sequences[state], nshots=nshots)
            for qubit in qubits:
                r = results[ro_pulses[qubit].serial].raw
                r.update(
                    {
                        "power[dBm]": [platform.get_lo_twpa_power(qubit)] * nshots,
                        "delta_power[dBm]": [power] * nshots,
                        "qubit": [qubit] * nshots,
                        "iteration": np.arange(nshots),
                        "state": [state] * nshots,
                    }
                )
                data.add_data_from_dict(r)

        # finally, save the remaining data and the fits
        yield data
        yield ro_optimization_fit(data, "delta_power", "state", "qubit", "iteration")
