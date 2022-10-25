# -*- coding: utf-8 -*-
import pdb
from cmath import exp
from copy import deepcopy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from qibo import gates, get_backend, models
from qibo.noise import NoiseModel, PauliError, ResetError, ThermalRelaxationError
from scipy.optimize import curve_fit

from qibocal import plots
from qibocal.data import Data
from qibocal.decorators import plot


def gellmann(j, k, d):
    """Returns a generalized Gell-Mann matrix of dimension d.
    According to the convention in *Bloch Vectors for Qubits* by
    Bertlmann and Krammer (2008),
    https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices#Construction
    Taken from Jonathan Gross. Revision 449580a1
    Parameters
    ----------
    j : int
        First index for generalized Gell-Mann matrix
    k : int
        Second index for generalized Gell-Mann matrix
    d : int
        Dimension of the generalized Gell-Mann matrix
    Returns
    -------
    A genereralized Gell-Mann matrix : np.ndarray
    """
    # Check the indices 'j' and 'k.
    if j > k:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = 1
        gjkd[k - 1][j - 1] = 1
    elif k > j:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = -1.0j
        gjkd[k - 1][j - 1] = 1.0j
    elif j == k and j < d:
        gjkd = np.sqrt(2 / (j * (j + 1))) * np.diag(
            [
                1 + 0.0j if n <= j else (-j + 0.0j if n == (j + 1) else 0 + 0.0j)
                for n in range(1, d + 1)
            ]
        )
    else:
        # Identity matrix
        gjkd = np.diag([1 + 0.0j for n in range(1, d + 1)])
        # normalize such that trace(gjkd*gjkd) = 2
        gjkd = gjkd * np.sqrt(2 / d)

    return gjkd


def get_basis(dim):
    """Return a basis of orthogonal Hermitian operators
    in a Hilbert space of dimension d, with the identity element
    in the last place.
    Taken from Jonathan Gross. Revision 449580a1
    Parameters
    ----------
    dim : int
        The amount of matrix basis elements and which
        dimension the matrices have.
    """
    return [gellmann(j, k, dim) for j, k in product(range(1, dim + 1), repeat=2)]


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

pauli = [np.eye(2) / np.sqrt(2), X / np.sqrt(2), Y / np.sqrt(2), Z / np.sqrt(2)]


def liouville_representation_errorchannel(error_channel, **kwargs):
    """For single qubit error channels only."""
    # For single qubit the dimension is two.
    dim = 2
    if error_channel.channel.__name__ == "PauliNoiseChannel":
        flipprobs = error_channel.options
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        def acts(gmatrix):
            return (
                (1 - flipprobs[0] - flipprobs[1] - flipprobs[2]) * gmatrix
                + flipprobs[0] * X @ gmatrix @ X
                + flipprobs[1] * Y @ gmatrix @ Y
                + flipprobs[2] * Z @ gmatrix @ Z
            )

    return np.array(
        [[np.trace(p2.conj().T @ acts(p1)) for p1 in pauli] for p2 in pauli]
    )


def effective_depol(error_channel, **kwargs):
    """ """
    liouvillerep = liouville_representation_errorchannel(error_channel)
    d = int(np.sqrt(len(liouvillerep)))
    depolp = ((np.trace(liouvillerep) + d) / (d + 1) - 1) / (d - 1)
    return depolp


class UIRS:
    """
    Uniform Independent Random Sequence
    """

    def __init__(self, nqubits, **kwargs):
        """ """
        self.nqubits = nqubits
        self.gate_generator = None
        self.invert = kwargs.get("invert", False)
        self.noisemodel = kwargs.get("noisemodel", None)
        # self.measurement = kwargs.get(
        #     'measurement', gates.M(*range(nqubits)))
        # CHANGED THAT
        self.measurement = kwargs.get("measurement", gates.M(2))

    def __call__(self, sequence_length):
        """For generating a sequence of circuits the object itself has to be
        called and the length of the sequence specified.

        Parameters
        ---------
        length : int
            How many circuits are created and put together for the sequence

        Return
        ------
        : models.Circuit
            An object which is executable as a simulation or on hardware
        """
        # Initiate the empty circuit from qibo with 'self.nqubits' many qubits.
        # circuit = models.Circuit(self.nqubits)
        circuit = models.Circuit(5)
        # Iterate over the sequence length.
        for _ in range(sequence_length):
            # Use the attribute to generate gates. This attribute can differ
            # for different classes since this encodes different gat sets.
            # For every loop add the generated matrix to the list of circuits.
            circuit.add(self.gate_generator())
        # For the standard randomized benchmarking scheme this is useful but
        # can also be ommitted and be done in the classical postprocessing.
        if self.invert:
            # FIXME ask Andrea: Something is not right with the inversion.
            # Invert all the already added circuits, multiply them with each
            # other and add as a new circuit to the list.
            # TODO changed fusion gate calculation by hand.
            # CHANGED THAT
            # circuit.add(gates.Unitary(circuit.invert().fuse().queue[0].matrix,0))
            circuit.add(gates.Unitary(circuit.invert().fuse().queue[0].matrix, 2))
        #     print(circuit.unitary())
        #     circuit.add(circuit.invert().fuse().queue[0])
        #     print(circuit.unitary())
        circuit.add(self.measurement)
        # For a simulation a noise model has to be added
        if self.noisemodel is None or not self.noisemodel:
            # No noise model means normally the circuit will be executed
            # on hardware.
            yield circuit
        else:
            # The noise stored in the attribute is applyed and generates
            # a new circuit with the wanted noise.
            noisy_circuit = self.noisemodel.apply(circuit)
            yield noisy_circuit


class UIRSOnequbitcliffords(UIRS):
    """
    # TODO optimize the Clifford drawing
    """

    # To not define the parameters for one qubit Cliffords every time a
    # new qubits is drawn define the parameters as global variable.
    global onequbit_clifford_params
    onequbit_clifford_params = [
        (0, 0, 0, 0),
        (np.pi, 1, 0, 0),
        (np.pi, 0, 1, 0),
        (np.pi, 0, 0, 1),
        (np.pi / 2, 1, 0, 0),
        (-np.pi / 2, 1, 0, 0),
        (np.pi / 2, 0, 1, 0),
        (-np.pi / 2, 0, 1, 0),
        (np.pi / 2, 0, 0, 1),
        (-np.pi / 2, 0, 0, 1),
        (np.pi, 1 / np.sqrt(2), 1 / np.sqrt(2), 0),
        (np.pi, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)),
        (np.pi, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)),
        (np.pi, -1 / np.sqrt(2), 1 / np.sqrt(2), 0),
        (np.pi, 1 / np.sqrt(2), 0, -1 / np.sqrt(2)),
        (np.pi, 0, -1 / np.sqrt(2), 1 / np.sqrt(2)),
        (2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
        (-2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
        (2 * np.pi / 3, -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
        (-2 * np.pi / 3, -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
        (2 * np.pi / 3, 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)),
        (-2 * np.pi / 3, 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)),
        (2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)),
        (-2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)),
    ]

    def __init__(self, nqubits, **kwargs):
        """ """
        super().__init__(nqubits, **kwargs)
        # Overwrite the gate generator attribute from the motherclass with
        # the class specific gneerator
        self.gate_generator = self.onequbit_clifford

    def Rn(self, theta=0, nx=0, ny=0, nz=0):
        """ """
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
        # CHANGED THAT
        # return gates.Unitary(matrix, 0)
        return gates.Unitary(matrix, 2)

    def onequbit_clifford(self, seed=None):
        """ """
        if seed is not None:
            backend = get_backend()
            backend.set_seed(seed)

        return self.Rn(
            *onequbit_clifford_params[np.random.randint(len(onequbit_clifford_params))]
        )


class UIRSNqubitscliffords(UIRSOnequbitcliffords):
    """ """

    def __init__(self, nqubits, **kwargs):
        """ """
        super().__init__(nqubits, **kwargs)
        self.gate_generator = self.nqubit_clifford

    def Rn(self, qubit, theta=0, nx=0, ny=0, nz=0):
        """ """
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
        return gates.Unitary(matrix, qubit)

    def nqubit_clifford(self):
        """ """
        circuit = models.Circuit(self.nqubits)

        for count in range(self.nqubits):
            circuit.add(
                self.Rn(
                    count,
                    *onequbit_clifford_params[
                        np.random.randint(len(onequbit_clifford_params))
                    ]
                )
            )
        qubitlist = list(np.arange(0, self.nqubits))
        blub = gates.Unitary(circuit.fuse().unitary(), *qubitlist)
        return blub


class Shadow:
    """Saves gate-set shadows, e.g. POVM measurement outcome and
    the corresponding gates leading to the state which was measured.

    For convenience the sequence length for the corresponding gates is stored
    along with the used set of POVMs.
    # TODO ask Ingo: What defines an outcome and what POVMs should we use?
    """

    def __init__(self, qubits, povms, sequence_lengths, runs, nshots):
        """Initiate three lists, one for gates, one for the corresponding
        outcomes and one for the corresponding sequence length of the gate.
        """
        self.povms = povms
        self.qubits = qubits
        self.circuit_list = []
        self.outcome_list = []
        self.samples_list = []
        self.probabilities_list = []
        self.sequence_lengths = sequence_lengths
        self.runs = runs
        self.nshots = nshots

    def append(self, circuit, outcome):
        """Update the shadow by appending the used circuits and the corresponding
        outcomes.
        """
        # Store the circuits.
        self.circuit_list.append(circuit)
        # Convert to list to check shape better.
        outcome = np.array(outcome)
        # Check the outcome type and shape. If it as as many entries as there
        # is shots for the experiment, is must be samples (and if it is a two
        # dimensional array like [[0],[0],...[1]])
        if len(outcome.shape) == 2 and outcome.shape[0] == self.nshots:
            # Flatten the outcome list and append it.
            self.samples_list.append(np.array(outcome).flatten())
        elif len(outcome.shape) == 1 and len(outcome) == int(2**self.qubits):
            # This must be the probabilities for the qubits.
            self.probabilities_list.append(np.array(outcome))
        else:
            raise ValueError("The given list doesnt have the right dimension.")

    @property
    def samples(self):
        """ """
        samples = np.array(self.samples_list)
        N = len(self.sequence_lengths)
        samples = samples.reshape((self.runs, N, self.nshots, self.qubits))
        return samples

    @property
    def probabilities(self):
        """ """
        if self.povms == "Ground State":
            if self.samples_list:
                return 1 - np.sum(self.samples, axis=2) / float(self.samples.shape[-1])
            elif self.probabilities_list:
                return np.array(self.probabilities_list)[:, 0].reshape(self.runs, -1)
        elif self.povms == "Basis Measurement":
            if self.probabilities_list:
                return np.array(self.probabilities_list).reshape(
                    self.runs, -1, int(2**self.qubits)
                )
            elif self.samples_list:
                return np.average(self.samples, axis=2)


def experimental_protocol(circuit_generator, myshadow, inject_noise=False, **kwargs):
    """

    Always the same, takes sequences of circuits of a certain gate set,
    applies them to |0><0|, followed by a measurement specified by a POVM
    {Ex}x.
    Saves the sequences of circuits along with the outcomes and the used
    POVMS for the outcomes.

    Paremeters
    ----------
    circuit_generator : UIRS or childclass object
        An object which can generate circuits drawn uniformly and
        independently at random from a specific gate set with a given
        sequence length.
    myshadow : Shadow object
        Stores gates and the corresponding outcomes, sequence length and
        povms
    runs : int
        The amount of loops over the whole protocol for a more meaningful
        statistics
    """
    # TODO ask Andrea: How to set the backend and platform nicely.
    # I want it actually linked to the run card.
    # TODO ask Andrea: How does qibolab knows which parameters to take for
    # my experiment?
    myshadow = deepcopy(myshadow)
    # Get the sequences from 'myshadow' object.
    sequence_lengths = myshadow.sequence_lengths
    runs = myshadow.runs
    # Store the circuit generator just to make the experiment repeatable.
    myshadow.cirucuit_generator = circuit_generator
    if inject_noise:
        pauli = PauliError(*inject_noise)
        noise = NoiseModel()
        noise.add(pauli, gates.Unitary)
    # Loop 'runs' many times over the whole protocol.
    for _ in range(runs):
        # Go through every sequence in the protocol.
        for m in sequence_lengths:
            # Generate the right circuits bunched to one gate.
            circuit = next(circuit_generator(m))
            if inject_noise:
                noisy_circuit = noise.apply(circuit)
                executed = noisy_circuit(nshots=kwargs.get("nshots"))
            else:
                # Execute the qibo circuit.
                executed = circuit(nshots=kwargs.get("nshots"))
            # TODO this will be changed in the future.
            # Also, this does not work for quantum hardware.
            try:
                # Get the samples from the executed gate. It should be an
                # object filled with as many integers as used shots.
                # outcome = executed.probabilities()
                outcome = executed.samples()
                probs = np.abs(executed.execution_result) ** 2
                print(outcome)
                print(probs)
                myshadow.probabilities_list.append(probs)
            except:
                # Getting the samples is not possible, hence the probabilities
                # have to be stored.
                outcome = executed.probabilities()
                print(outcome)
                print(executed.samples())
            # Store the samples.
            myshadow.append(circuit, outcome)
            # Store everything.
    return myshadow


def test_hardware(platform, qubit: list, nshots):
    circuit = models.Circuit(5)
    circuit.add(gates.H(2))
    circuit.add(gates.M(2))
    # circuit.add(gates.X(3))
    # circuit.add(gates.M(3))
    print(circuit.draw())
    executed = circuit()
    print(executed.probabilities())
    print(executed.execution_result)
    print(execute.samples())


@plot("Test Standard RB", plots.standard_rb_plot)
def standard_rb(
    platform, qubit: list, generator_name, sequence_lengths, runs, nshots, inject_noise
):
    # Define the data object.
    # Use the sequence_lengths list as labels for the data
    data1 = Data("standardrb", quantities=list(sequence_lengths))
    # Generate the circuits
    measurement_type = "Ground State"
    myshadow = Shadow(len(qubit), measurement_type, sequence_lengths, runs, nshots)
    if generator_name == "UIRSOnequbitcliffords":
        mygenerator = UIRSOnequbitcliffords(1, invert=True, noisemodel=False)
    else:
        raise ValueError("This generator is not implemented.")
    myshadow = experimental_protocol(
        mygenerator, myshadow, inject_noise=inject_noise, nshots=nshots
    )
    for count in range(runs):
        data1.add(
            {
                sequence_lengths[i]: myshadow.probabilities[count][i]
                for i in range(len(sequence_lengths))
            }
        )
    # pm = np.sum(myshadow.probabilities, axis=0)/runs
    # print(sequence_lengths)
    # print(pm)
    # data.add({'survival_probabilities':pm,
    #     'sequence_lengths':sequence_lengths})
    yield data1
    pauli = PauliError(*inject_noise)
    noise = NoiseModel()
    noise.add(pauli, gates.Unitary)
    data2 = Data("effectivedepol", quantities=["effective_depol"])
    data2.add({"effective_depol": effective_depol(pauli)})
    yield data2


# @plot("Test Standard RB", plots.standard_rb_plot)
def standard_rb_test(
    platform, qubit: list, generator_name, sequence_lengths, runs, nshots, inject_noise
):
    # Define the data object.
    # Use the sequence_lengths list as labels for the data
    data1 = Data("standardrb_data", quantities=list(sequence_lengths))
    data2 = Data("standardrb_circuits", quantities=list(sequence_lengths))
    # Generate the circuits
    measurement_type = "Ground State"
    myshadow = Shadow(len(qubit), measurement_type, sequence_lengths, runs, nshots)
    if generator_name == "UIRSOnequbitcliffords":
        mygenerator = UIRSOnequbitcliffords(1, invert=True, noisemodel=False)
    else:
        raise ValueError("This generator is not implemented.")
    myshadow = experimental_protocol(
        mygenerator, myshadow, inject_noise=inject_noise, nshots=nshots
    )
    for count in range(runs):
        data1.add(
            {
                sequence_lengths[i]: myshadow.probabilities[count][i]
                for i in range(len(sequence_lengths))
            }
        )
        data2.add(
            {
                sequence_lengths[i]: myshadow.circuit_list[
                    count * len(sequence_lengths) + i
                ]
                for i in range(len(sequence_lengths))
            }
        )
    # pm = np.sum(myshadow.probabilities, axis=0)/runs
    # print(sequence_lengths)
    # print(pm)
    # data.add({'survival_probabilities':pm,
    #     'sequence_lengths':sequence_lengths})
    yield data1
    yield data2
    pauli = PauliError(*inject_noise)
    noise = NoiseModel()
    noise.add(pauli, gates.Unitary)
    data3 = Data("effectivedepol", quantities=["effective_depol"])
    data3.add({"effective_depol": effective_depol(pauli)})
    yield data3


@plot("Test Standard RB", plots.standard_rb_plot)
def filtered_rb(
    platform, qubit: list, generator_name, sequence_lengths, runs, nshots, inject_noise
):
    # Define the data object.
    # Use the sequence_lengths list as rows (labels) for the data.
    data1 = Data("standardrb", quantities=list(sequence_lengths))
    measurement_type = "Ground State"
    myshadow = Shadow(len(qubit), measurement_type, sequence_lengths, runs, nshots)
    # Get the circuit generator.
    if generator_name == "UIRSOnequbitcliffords":
        mygenerator = UIRSOnequbitcliffords(1, invert=False, noisemodel=False)
    else:
        raise ValueError("This generator is not implemented.")
    # Perform the experiment.
    myshadow = experimental_protocol(
        mygenerator, myshadow, inject_noise=inject_noise, nshots=nshots
    )
    # Calculate the postprocessed data, here meaning the filter values.
    d = 2
    amount_sequences = len(sequence_lengths)
    filterslist = []
    if False:  # single shot
        for count in range(runs):
            for m in range(amount_sequences):
                mycircuit = myshadow.circuit_list[count * amount_sequences + m]
                executed_circuit = mycircuit(nshots=nshots)
                filterf = 0
                talpha, tbeta = executed_circuit.probabilities()
                alpha, beta = (
                    myshadow.probabilities[count, m],
                    1 - myshadow.probabilities[count, m],
                )
                for shot in range(nshots):
                    outcome = myshadow.samples_list[count * amount_sequences + m][shot]
                    prob = executed_circuit.probabilities()[int(outcome)]
                    filterf += (d + 1) * (np.abs(prob) - 1 / d)
                filterslist.append(filterf / nshots)
        filtersarray = np.array(filterslist).reshape(runs, amount_sequences)
    else:  # probabilities
        for count in range(runs):
            for m in range(amount_sequences):
                mycircuit = myshadow.circuit_list[count * amount_sequences + m]
                executed_circuit = mycircuit(nshots=nshots)
                filterf = 0
                talpha, tbeta = executed_circuit.probabilities()
                alpha = np.sqrt(myshadow.probabilities[count, m])
                beta = np.sqrt(1 - myshadow.probabilities[count, m])
                talpha, tbeta = np.sqrt(executed_circuit.probabilities()[0])
                filterf = (d + 1) * (np.abs(alpha * talpha + beta * tbeta) ** 2 - 1 / d)
                filterslist.append(filterf)
        filtersarray = np.array(filterslist).reshape(runs, amount_sequences)
    for count in range(runs):
        data1.add(
            {
                sequence_lengths[i]: filtersarray[count][i]
                for i in range(len(sequence_lengths))
            }
        )

    yield data1
    if inject_noise:
        pauli = PauliError(*inject_noise)
        noise = NoiseModel()
        noise.add(pauli, gates.Unitary)
        effective = effective_depol(pauli)
    else:
        effective = 0
    data2 = Data("effectivedepol", quantities=["effective_depol"])
    data2.add({"effective_depol": effective})
    yield data2


@plot("Test Standard RB", plots.standard_rb_plot)
def sim_rb(
    platform, qubit: list, generator_name, sequence_lengths, runs, nshots, inject_noise
):
    # Define the data object.
    # Use the sequence_lengths list as rows (labels) for the data.
    data1 = Data("standardrb", quantities=list(sequence_lengths))
    measurement_type = "Basis Measurement"
    myshadow = Shadow(len(qubit), measurement_type, sequence_lengths, runs, nshots)
    # Get the circuit generator.
    if generator_name == "UIRSOnequbitcliffords":
        mygenerator = UIRSOnequbitcliffords(1, invert=False, noisemodel=False)
    elif generator_name == "UIRSNqubitscliffords":
        mygenerator = UIRSNqubitscliffords(len(qubit), invert=False, noisemodel=False)
    else:
        raise ValueError("This generator is not implemented.")
    # Perform the experiment.
    myshadow = experimental_protocol(
        mygenerator, myshadow, inject_noise=inject_noise, nshots=nshots
    )
    # Calculate the postprocessed data, here meaning the filter values.
    d = 2 ** len(qubit)
    amount_sequences = len(sequence_lengths)
    filterslist = []
    if False:  # single shot
        for count in range(runs):
            for m in range(amount_sequences):
                mycircuit = myshadow.circuit_list[count * amount_sequences + m]
                executed_circuit = mycircuit(nshots=nshots)
                filterf = 0
                probs_noerror = np.abs(executed_circuit.execution_result) ** 2
                probs_error = myshadow.probabilities[count, m]
                for shot in range(nshots):
                    outcome = myshadow.samples_list[count * amount_sequences + m][shot]
                    prob = executed_circuit.probabilities()[int(outcome)]
                    filterf += (d + 1) * (np.abs(prob) - 1 / d)
                filterslist.append(filterf / nshots)
        filtersarray = np.array(filterslist).reshape(runs, amount_sequences)
    else:  # probabilities
        for count in range(runs):
            for m in range(amount_sequences):
                mycircuit = myshadow.circuit_list[count * amount_sequences + m]
                executed_circuit = mycircuit(nshots=nshots)
                filterf = 0
                probs_noerror = np.abs(executed_circuit.execution_result) ** 2
                probs_error = myshadow.probabilities[count, m]
                a = (
                    np.abs(
                        np.sqrt(probs_noerror[0] + probs_noerror[1])
                        * np.sqrt(probs_error[0] + probs_error[1])
                        + np.sqrt(probs_noerror[2] + probs_noerror[3])
                        * np.sqrt(probs_error[2] + probs_error[3])
                    )
                    ** 2
                )
                b = (
                    np.abs(
                        np.sqrt(probs_noerror[0] + probs_noerror[2])
                        * np.sqrt(probs_error[0] + probs_error[2])
                        + np.sqrt(probs_noerror[1] + probs_noerror[3])
                        * np.sqrt(probs_error[1] + probs_error[3])
                    )
                    ** 2
                )
                filterf = (2 + 1) * a * (b - 1 / 2)
                filterslist.append(filterf)
        filtersarray = np.array(filterslist).reshape(runs, amount_sequences)
    for count in range(runs):
        data1.add(
            {
                sequence_lengths[i]: filtersarray[count][i]
                for i in range(len(sequence_lengths))
            }
        )

    yield data1
    if inject_noise:
        pauli = PauliError(*inject_noise)
        noise = NoiseModel()
        noise.add(pauli, gates.Unitary)
        effective = effective_depol(pauli)
    else:
        effective = 0
    data2 = Data("effectivedepol", quantities=["effective_depol"])
    data2.add({"effective_depol": effective})
    yield data2
