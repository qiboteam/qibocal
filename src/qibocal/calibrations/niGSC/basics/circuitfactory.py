from __future__ import annotations

import abc
from copy import deepcopy

import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibo.quantum_info.random_ensembles import random_clifford

from qibocal.calibrations.niGSC.basics.utils import ONEQ_GATES
from qibocal.config import raise_error


class CircuitFactory:
    """Iterator object, when called a random circuit with wanted gate
    distribution is created.
    """

    def __init__(
        self, nqubits: int, depths: list | np.ndarray | int, qubits: list = []
    ) -> None:
        self.nqubits = nqubits if nqubits is not None else len(qubits)
        self.qubits = qubits if qubits else list(range(nqubits))
        if isinstance(depths, int):
            depths = [depths]
        self.depths = depths
        self.name = "Abstract"

    def __len__(self):
        return len(self.depths)

    def __iter__(self) -> CircuitFactory:
        self.n = 0
        return self

    def __next__(self) -> Circuit:
        """Build a ``Circuit`` object with wanted gate distribution.

        Embeds the circuit if the number of qubits ``self.nqubits`` is bigger than
        the amount of qubits indicated in ``self.qubits``.

        Raises:
            StopIteration: When the generator comes to an end.

        Returns:
            Circuit: With specific gate distribution.
        """

        # Check if the stop critarion is met.
        if self.n >= len(self.depths):
            raise StopIteration

        circuit = self.build_circuit(self.depths[self.n])
        self.n += 1
        # Distribute the circuit onto the given support.
        circuit_init_kwargs = deepcopy(circuit.init_kwargs)
        circuit_init_kwargs["nqubits"] = self.nqubits
        bigcircuit = Circuit(**circuit_init_kwargs)
        bigcircuit.add(circuit.on_qubits(*self.qubits))
        return bigcircuit

    def build_circuit(self, depth: int) -> Circuit:
        """Initiate a ``qibo.models.Circuit`` object and fill it with the wanted gates.

        Which gates are wanted is encoded in ``self.gates_layer()``.
        Add a measurement gate for every qubit.

        Args:
            depth (int): How many layers there are in the circuit.

        Returns:
            Circuit: the circuit with ``depth`` many layers.
        """
        # Initiate the ``Circuit`` object with the amount of active qubits.
        circuit = Circuit(len(self.qubits))
        # Go throught the depth/layers of the circuit and add gate layers.
        for _ in range(depth):
            circuit.add(self.gate_layer())
        # Add a ``Measurement`` gate for every qubit.
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit

    @abc.abstractmethod
    def gate_layer(self):
        """This method has to be overwritten by the inheriting child class."""
        raise_error(NotImplementedError)


class Qibo1qGatesFactory(CircuitFactory):
    """When called creates a random circuit build out of 1-qubit non-parameterized
    qibo gates.
    """

    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits)
        self.name = "Qibo1qGates"

    def gate_layer(self):
        """Build a circuit out of random 1-qubit qibo gates.

        Returns:
            (list) filled with random 1 qubit qibo gates
        """
        gates_list = []
        # Draw the random indices for the list where the names of the 1-qubit
        # non-parameterized gates are stored.
        for count, rint in enumerate(
            np.random.randint(0, len(ONEQ_GATES), size=len(self.qubits))
        ):
            # Load the random gate.
            rand_gate = getattr(gates, ONEQ_GATES[rint])
            # Append the random gate initialized with the qubit is should act on.
            gates_list.append(rand_gate(count))
        return gates_list


class SingleCliffordsFactory(CircuitFactory):
    """Creates circuits filled with random  single qubit Clifford gates for
    each active qubit.
    """

    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits)
        self.name = "SingleCliffords"

    def gate_layer(self) -> list:
        """Use the ``qibo.quantum_info`` module to draw as many random clifford
        unitaries as there are (active) qubits, make unitary gates with them.

        Returns:
            (list) filled with ``qibo.gates.Unitary``: the simulatanous 1q-Clifford gates.
        """

        gates_list = []
        # Make sure the shape is suitable for iterating over the Clifford matrices returned
        # by the ``random_clifford`` function.
        random_cliffords = random_clifford(self.qubits).reshape(len(self.qubits), 2, 2)
        # Make gates out of the unitary matrices.
        for count, rand_cliff in enumerate(random_cliffords):
            # Build the gate with the random Clifford matrix, let is act on the right qubit.
            gates_list.append(gates.Unitary(rand_cliff, count))
        return gates_list
