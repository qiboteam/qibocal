
from qibo import gates, models
import numpy as np
from qibocal.calibrations.protocols.utils import ONEQUBIT_CLIFFORD_PARAMS

class Circuitfactory(): 
    def __init__(self, qubits: list, sequences: list, runs: int) -> None:
        self.qubits = qubits
        self.sequences = sequences
        self.runs = runs
    def __iter__(self) -> None:
        self.n = 0
        return self
    def __next__(self) -> None:
        if self.n > self.runs * len(self.sequences):
            raise StopIteration
        else:
            circuit = self.build_circuit(
                self.sequences[self.n % len(self.sequences)])
            self.n += 1
            return circuit
    def build_circuit(self, depth: int):
        raise NotImplementedError

class SingleCliffordsFactory(Circuitfactory):
    def __init__(self, qubits: list, sequences: list, runs: int) -> None:
        super().__init__(qubits, sequences, runs)
    
    def circuit_generator(self):
        for _ in range(self.runs):
            for depth in self.sequences:
                yield self.build_circuit(depth)

    def build_circuit(self, depth: int):
        circuit = models.Circuit(len(self.qubits))
        for _ in range(depth):
            circuit.add(self.gate())
        circuit.add(gates.M(*self.qubits))
        return circuit

    def clifford_unitary(
        self, theta: float = 0, nx: float = 0, ny: float = 0, nz: float = 0
    ) -> np.ndarray:
        """Four given parameters are used to build one Clifford unitary.

        Args:
            theta (float) : An angle
            nx (float) : prefactor
            ny (float) : prefactor
            nz (float) : prefactor

        Returns:
            ``qibo.gates.Unitary`` with the drawn matrix as unitary.
        """
        matrix = np.array(
            [
                [
                    np.cos(theta / 2) - 1.0j * nz * np.sin(theta / 2),
                    -ny * np.sin(theta / 2) - 1.0j * nx * np.sin(theta / 2),
                ],
                [
                    ny * np.sin(theta / 2) - 1.0j * nx * np.sin(theta / 2),
                    np.cos(theta / 2) + 1.0j * nz * np.sin(theta / 2),
                ],
            ]
        )
        return matrix

    def gate(self) -> gates.Unitary:
        """Draws the parameters and builds the gate.

        Args:
            seed (int): optional, set the starting seed for random
                        number generator.
        Returns:
            ``qibo.gates.Unitary``: the simulatanous Clifford gates
        """
        # There are this many different Clifford matrices.
        amount = len(ONEQUBIT_CLIFFORD_PARAMS)
        # Initiate the matrix to start the kronecker (tensor) product.
        unitary = self.clifford_unitary(
            *ONEQUBIT_CLIFFORD_PARAMS[np.random.randint(amount)]
        )
        # Choose as many random integers between 0 and 23 as there are used
        # qubits. Get the clifford parameters and build the unitary.
        for rint in np.random.randint(0, amount, size=len(self.qubits) - 1):
            # Build the random Clifford matrix and take the tensor product
            # with the matrix before.
            unitary = np.kron(
                self.clifford_unitary(*ONEQUBIT_CLIFFORD_PARAMS[rint]), unitary
            )
        # Make a unitary gate out of 'unitary' for the qubits.
        return gates.Unitary(unitary, *self.qubits)

class Experiment():
    def __init__(self) -> None:
        pass
    def execute(self) -> None:
        pass
    def single_execute(self) -> None:
        pass


