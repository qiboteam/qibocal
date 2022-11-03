# -*- coding: utf-8 -*-
import pdb
from ctypes import Union

import numpy as np
from qibo import gates, get_backend, models


from qibocal.calibrations.protocols.utils import onequbit_clifford_params


class Generator:
    """
    Uniform Independent Random Sequence
    """

    def __init__(self, qubits: list, act_on: int = None, **kwargs) -> None:
        """ """
        # Check the type of the variable 'qubits', different versions of
        # qibocal require different types in the WIP version.
        if type(qubits) == int:
            # Make it list out of the interger given.
            self.qubits = [x for x in range(qubits)]
        elif type(qubits) == list or type(qubits) == np.ndarray:
            self.qubits = qubits
        else:
            raise ValueError("Wrong type of qubits given.")
        self.gate_generator = None
        # For standard RB the inverse is needed, but it can be costly to
        # calculate hence filtered RB and other protocols don't need it.
        self.invert = kwargs.get("invert", False)
        # Sometimes not all qubits should be used, only one.
        if act_on:
            self.used_qubits = [act_on]
        else:
            self.used_qubits = self.qubits
        # Every used qubit should be measured in the end (basis measurement).
        self.measurement = kwargs.get("measurement", gates.M(*self.used_qubits))

    def __call__(self, sequence_length: list):
        """For generating a sequence of circuits the object itself
        has to be called and the length of the sequence specified.

        Args:
            length (int) : How many circuits are created and put
                           together for the sequence.

        Returns:
        (list): with the minimal representation of the circuit.
        ``qibo.models.Circuit``: object which is executable as a
                                 simulation or on hardware
        """
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = models.Circuit(len(self.qubits))
        # Iterate over the sequence length.
        for _ in range(sequence_length):
            # Use the attribute to generate gates. This attribute can
            # differ for different classes since this encodes different
            # gat sets. For every loop retrieve the gate.
            gate = self.gate_generator()
            # Add the generated unitary gate to the list of circuits.
            circuit.add(gate)
        # For the standard randomized benchmarking scheme this is
        # useful but can also be ommitted and be done in the classical
        # postprocessing.
        if self.invert:
            # FIXME changed fusion gate calculation by hand since the
            # inbuilt function does not work.
            # Calculate the inversion matrix.
            inversion_unitary = circuit.invert().fuse().queue[0].matrix
            # Add it as a unitary gate to the circuit.
            circuit.add(gates.Unitary(inversion_unitary, *self.used_qubits))
        circuit.add(self.measurement)
        # No noise model added, for a simulation either the platform
        # introduces the errors or the error gates will be added
        # before execution.
        yield circuit


class GeneratorOnequbitcliffords(Generator):
    """
    TODO optimize the Clifford drawing
    """

    def __init__(self, qubits, **kwargs):
        """ """
        super().__init__(qubits, **kwargs)
        # Overwrite the gate generator attribute from the motherclass with
        # the class specific generator.
        self.gate_generator = self.nqubit_clifford

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

    def nqubit_clifford(self, seed: int = None) -> gates.Unitary:
        """Draws the parameters and builds the gate.

        Args:
            seed (int): optional, set the starting seed for random
                        number generator.
        Returns:
            ``qibo.gates.Unitary``: the simulatanous Clifford gates
        """
        # Make it possible to set the seed for the random number generator.
        if seed is not None:
            # Get the backend and set the seed.
            backend = get_backend()
            backend.set_seed(seed)
        # There are this many different Clifford matrices.
        amount = len(onequbit_clifford_params)
        # Initiate the matrix to start the kronecker (tensor) product.
        unitary = self.clifford_unitary(
            *onequbit_clifford_params[np.random.randint(amount)]
        )
        # Choose as many random integers between 0 and 23 as there are
        # used qubits.
        # Get the clifford parameters and build the unitary.
        for rint in np.random.randint(0, amount, size=len(self.used_qubits) - 1):
            # Build the random Clifford matrix and take the tensor product
            # with the matrix before.
            unitary = np.kron(
                self.clifford_unitary(*onequbit_clifford_params[rint]), unitary
            )
        # Make a unitary gate out of 'unitary' for the qubits.
        return gates.Unitary(unitary, *self.used_qubits)
