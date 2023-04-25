import copy
import time

import numpy as np
import pandas as pd
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal import plots
from qibocal.data import Data, DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import calibrate_qubit_states_fit, ro_optimization_fit


@plot("Qubit States", plots.ro_frequency)
def ro_frequency(
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
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
    )

    # execute the first pulse sequence
    state0_results = platform.sweep(
        state0_sequence, frequency_sweeper, nshots=nshots, average=False
    )

    for qubit in qubits:
        r = {
            k: v.ravel() for k, v in state0_results[ro_pulses[qubit].serial].raw.items()
        }
        r.update(
            {
                "frequency[Hz]": np.repeat(
                    np.vstack(delta_frequency_range).T,
                    len(np.arange(nshots)),
                    axis=0,
                ).flatten()
                + ro_pulses[qubit].frequency,
                "delta_frequency[Hz]": np.repeat(
                    np.vstack(delta_frequency_range).T,
                    len(np.arange(nshots)),
                    axis=0,
                ).flatten(),
                "qubit": [qubit] * nshots * len(delta_frequency_range),
                "iteration": np.repeat(
                    np.vstack(np.arange(nshots)).T,
                    len(delta_frequency_range),
                    axis=1,
                ).flatten(),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [0] * nshots * len(delta_frequency_range),
            }
        )
        data.add_data_from_dict(r)
    yield data

    # execute the second pulse sequence
    state1_results = platform.sweep(
        state1_sequence, frequency_sweeper, nshots=nshots, average=False
    )

    # retrieve and store the results for every qubit)
    for qubit in qubits:
        r = {
            k: v.ravel() for k, v in state1_results[ro_pulses[qubit].serial].raw.items()
        }
        r.update(
            {
                "frequency[Hz]": np.repeat(
                    np.vstack(delta_frequency_range).T,
                    len(np.arange(nshots)),
                    axis=0,
                ).flatten()
                + ro_pulses[qubit].frequency,
                "delta_frequency[Hz]": np.repeat(
                    np.vstack(delta_frequency_range).T,
                    len(np.arange(nshots)),
                    axis=0,
                ).flatten(),
                "qubit": [qubit] * nshots * len(delta_frequency_range),
                "iteration": np.repeat(
                    np.vstack(np.arange(nshots)).T,
                    len(delta_frequency_range),
                    axis=1,
                ).flatten(),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [1] * nshots * len(delta_frequency_range),
            }
        )
        data.add_data_from_dict(r)

    # finally, save the remaining data and the fits
    yield data
    yield ro_optimization_fit(data, "delta_frequency")


@plot("Qubit States", plots.ro_amplitude)
def ro_amplitude(
    platform: AbstractPlatform,
    qubits: list,
    amplitude_factor_min: float,
    amplitude_factor_max: float,
    amplitude_factor_step: float,
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
        amplitude_factor_min, amplitude_factor_max, amplitude_factor_step
    )

    amplitude_sweeper = Sweeper(
        Parameter.amplitude,
        delta_amplitude_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
    )

    # execute the first pulse sequence
    state0_results = platform.sweep(
        state0_sequence, amplitude_sweeper, nshots=nshots, average=False
    )

    for qubit in qubits:
        r = {
            k: v.ravel() for k, v in state0_results[ro_pulses[qubit].serial].raw.items()
        }
        print(f" {qubit}: State0 amplitude: {ro_pulses[qubit].amplitude}")
        print(
            f" {qubit}: amplitude unique: {np.unique(np.repeat(np.vstack(delta_amplitude_range).T,len(np.arange(nshots)),axis=0).flatten()* ro_pulses[qubit].amplitude)}"
        )
        r.update(
            {
                "amplitude[dimensionless]": np.repeat(
                    np.vstack(delta_amplitude_range).T,
                    len(np.arange(nshots)),
                    axis=0,
                ).flatten()
                * ro_pulses[qubit].amplitude,
                "delta_amplitude[dimensionless]": np.repeat(
                    np.vstack(delta_amplitude_range).T,
                    len(np.arange(nshots)),
                    axis=0,
                ).flatten(),
                "qubit": [qubit] * nshots * len(delta_amplitude_range),
                "iteration": np.repeat(
                    np.vstack(np.arange(nshots)).T,
                    len(delta_amplitude_range),
                    axis=1,
                ).flatten(),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [0] * nshots * len(delta_amplitude_range),
            }
        )
        data.add_data_from_dict(r)
    yield data

    # execute the second pulse sequence
    state1_results = platform.sweep(
        state1_sequence, amplitude_sweeper, nshots=nshots, average=False
    )

    # retrieve and store the results for every qubit)
    for qubit in qubits:
        r = {
            k: v.ravel() for k, v in state1_results[ro_pulses[qubit].serial].raw.items()
        }
        print(f" {qubit}: State1 amplitude: {ro_pulses[qubit].amplitude}")
        print(
            f" {qubit}: amplitude unique: {np.unique(np.repeat(np.vstack(delta_amplitude_range).T,len(np.arange(nshots)),axis=0).flatten()* ro_pulses[qubit].amplitude)}"
        )
        r.update(
            {
                "amplitude[dimensionless]": np.repeat(
                    np.vstack(delta_amplitude_range).T,
                    len(np.arange(nshots)),
                    axis=0,
                ).flatten()
                * ro_pulses[qubit].amplitude,
                "delta_amplitude[dimensionless]": np.repeat(
                    np.vstack(delta_amplitude_range).T,
                    len(np.arange(nshots)),
                    axis=0,
                ).flatten(),
                "qubit": [qubit] * nshots * len(delta_amplitude_range),
                "iteration": np.repeat(
                    np.vstack(np.arange(nshots)).T,
                    len(delta_amplitude_range),
                    axis=1,
                ).flatten(),  # Might be the other way depending on how is result happening. Axis=0 gives 123123 and axis=1 gives 1112233
                "state": [1] * nshots * len(delta_amplitude_range),
            }
        )
        data.add_data_from_dict(r)

    # finally, save the remaining data and the fits
    yield data
    yield ro_optimization_fit(data, "delta_amplitude")
