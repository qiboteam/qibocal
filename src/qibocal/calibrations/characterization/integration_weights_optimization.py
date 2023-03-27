import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("Signal_0-1", plots.signal_0_1)
def integration_weights_optimization(
    platform: AbstractPlatform,
    qubits: dict,
    nshots,
    relaxation_time=None,
):
    """
    Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed
    for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (dict): Dict of target Qubit objects to perform the action
        nshots (int): number of times the pulse sequence will be repeated.
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Signal voltage mesurement in volts before demodulation
            - **iteration[dimensionless]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

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
        quantities={"weights": "dimensionless"},
        options=["qubit", "sample", "state"],
    )
    # execute the first pulse sequence
    # state0_results = platform.execute_pulse_sequence(state0_sequence, nshots=nshots, relaxation_time=relaxation_time, acquistion = "RAW")
    state0_results = platform.execute_pulse_sequence(
        state0_sequence, nshots=nshots, relaxation_time=relaxation_time
    )

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = state0_results[ro_pulse.serial].to_dict(average=False)
        state0 = r["i[V]"] + 1j * r["q[V]"]
        number_of_samples = len(r["MSR[V]"])
        r.update(
            {
                "qubit": [ro_pulse.qubit] * len(r["MSR[V]"]),
                "sample": np.arange(len(r["MSR[V]"])),
                "state": [0] * len(r["MSR[V]"]),
            }
        )
        data.add_data_from_dict(r)

    # execute the second pulse sequence
    # state1_results = platform.execute_pulse_sequence(state1_sequence, nshots=nshots, acquistion = "RAW")
    state1_results = platform.execute_pulse_sequence(state1_sequence, nshots=nshots)
    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = state1_results[ro_pulse.serial].to_dict(average=False)
        state1 = r["i[V]"] + 1j * r["q[V]"]
        r.update(
            {
                "qubit": [ro_pulse.qubit] * len(r["MSR[V]"]),
                "sample": np.arange(len(r["MSR[V]"])),
                "state": [1] * len(r["MSR[V]"]),
            }
        )
        data.add_data_from_dict(r)

    # finally, save the remaining data and the fits

    # np.conj to account the two phase-space evolutions of the readout state
    samples_kernel = np.conj(state1 - state0)
    # Remove nans
    samples_kernel = samples_kernel[~np.isnan(samples_kernel)]

    samples_kernel_origin = (
        samples_kernel - samples_kernel.real.min() - 1j * samples_kernel.imag.min()
    )  # origin offsetted
    samples_kernel_normalized = (
        samples_kernel_origin / np.abs(samples_kernel_origin).max()
    )  # normalized

    r = {}
    r.update(
        {
            "weights[dimensionless]": abs(samples_kernel_normalized),
            "qubit": [ro_pulse.qubit] * number_of_samples,
            "sample": np.arange(number_of_samples),
            "state": ["1-0"] * number_of_samples,
        }
    )
    data.add_data_from_dict(r)

    yield data

    np.save(
        "/home/admin/Juan/qibolab/src/qibolab/instruments/Optimal_weights_conj",
        samples_kernel_normalized,
    )
