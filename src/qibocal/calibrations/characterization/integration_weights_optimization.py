import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.platforms.platform import AcquisitionType
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


# Cant find VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences
@plot("Signal_0-1", plots.signal_0_1)
def integration_weights_optimization(
    platform: AbstractPlatform,
    qubits: dict,
    nshots,
    relaxation_time=None,
):
    """
    Method which implements the calculation of the optimal integration weights for a qubit given by amplifying the parts of the
    signal acquired which maximally distinguish between state 1 and 0.

    Args:
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): custom abstract platform on which we perform the calibration.
        qubits (dict): Dict of target Qubit objects to perform the action
        nshots (int) : Number of executions on hardware of the routine for averaging results
        relaxation_time (float): Wait time for the qubit to decohere back to the `gnd` state

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Signal voltage mesurement in volts before demodulation
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **weights**: Optimal integration weights for this case
            - **qubit**: The qubit being tested
            - **sample**: Sample number of the acquiered signal
            - **state**: State of the qubit

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
        state0_results = platform.execute_pulse_sequence(
            state0_sequence,
            nshots=nshots,
            relaxation_time=relaxation_time,
            acquisition_type=AcquisitionType.RAW,
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
        state1_results = platform.execute_pulse_sequence(
            state1_sequence,
            nshots=nshots,
            relaxation_time=relaxation_time,
            acquisition_type=AcquisitionType.RAW,
        )
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

        # Post-processing
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
        yield samples_kernel_normalized, "_qubit_" + str(qubit)
