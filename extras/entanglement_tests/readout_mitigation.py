import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibolab.pulses import PulseSequence
from utils import calculate_frequencies


class ReadoutErrorMitigation:
    def __init__(self, platform, nqubits, qubits, readout_error_model=(0.0, 0.0)):
        """Platform should be left None to default to simulation. nqubits should be the total
        number of qubits in the chip, while qubits are the qubits that are targetted.
        """
        self.platform = platform
        self.nqubits = nqubits
        self.qubits = qubits
        self.calibration_matrix = None
        self.rerr = readout_error_model

    def get_calibration_matrix(self, nshots=1024):
        """Self explanatory. Prepare states and measure in order to get the readout
        matrix for error correction.
        """
        nqubits = self.nqubits
        qubits = self.qubits

        nq = len(qubits)

        matrix = np.zeros((2**nq, 2**nq))

        platform = self.platform

        if nq != 2:
            raise ValueError("Only 2 qubits supported for now.")

        for i in range(2**nq):
            state = format(i, f"0{nq}b")
            if platform:
                sequence = PulseSequence()
                for q, bit in enumerate(state):
                    if bit == "1":
                        sequence.add(
                            platform.create_RX_pulse(
                                qubits[q], start=0, relative_phase=0
                            )
                        )
                measurement_start = sequence.finish
                for qubit in qubits:
                    MZ_pulse = platform.create_MZ_pulse(qubit, start=measurement_start)
                    sequence.add(MZ_pulse)
                results = platform.execute_pulse_sequence(sequence, nshots=nshots)
                freqs = calculate_frequencies(results[qubits[0]], results[qubits[1]])
            else:
                c = Circuit(nqubits)
                for q, bit in enumerate(state):
                    if bit == "1":
                        c.add(gates.X(qubits[q]))
                for qubit in qubits:
                    c.add(gates.M(qubit, p0=self.rerr[0], p1=self.rerr[1]))
                results = c(nshots=nshots)
                freqs = results.frequencies()

            column = np.zeros(2**nq)
            for key in freqs.keys():
                f = freqs[key]
                column[int(key, 2)] = f / nshots
            matrix[:, i] = column

        self.calibration_matrix = np.linalg.inv(matrix)

        return self.calibration_matrix

    def apply_readout_mitigation(self, frequencies, calibration_matrix=None):
        """Updates the frequencies of the input state with the mitigated ones obtained with `calibration_matrix`*`state.frequencies()`.

        Args:
                state (qibo.states.CircuitResult): Input state to be updated.
                calibration_matrix (np.ndarray): Calibration matrix for readout mitigation.

        Returns:
                qibo.states.CircuitResult : The input state with the updated frequencies.
        """
        qubits = self.qubits
        nqubits = self.nqubits
        nq = len(qubits)

        if calibration_matrix == None:
            if self.calibration_matrix is None:
                raise ValueError(
                    "Readout Mitigation Matrix has not been calibrated yet!"
                )
            else:
                calibration_matrix = self.calibration_matrix

        freq = np.zeros(2**nq)

        for k, v in frequencies.items():
            freq[int(k, 2)] = v

        freq = freq.reshape(-1, 1)
        new_freq = {}
        for i, val in enumerate(calibration_matrix @ freq):
            new_freq[format(i, f"0{nq}b")] = float(val)

        return new_freq
