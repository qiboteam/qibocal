# -*- coding: utf-8 -*-
from qibo import gates, models

from qcvv.data import Data


from cmath import exp
import numpy as np
from qibo import gates, get_backend, models
import pdb
from copy import deepcopy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from qibo.noise import PauliError, ThermalRelaxationError, NoiseModel, ResetError


def test(
    platform,
    qubit: list,
    nshots,
    points=1,
):
    mylist = list(map(str, [0, 5]))
    data = Data("test", quantities=mylist)
    data.add({f"{mylist[0]}":13., f"{mylist[1]}":1.})
    data.add({f"{mylist[0]}":14., f"{mylist[1]}":2.})
    # data = Data("test", quantities=["nshots", "probabilities"])
    # nqubits = len(qubit)
    # circuit = models.Circuit(nqubits)
    # circuit.add(gates.H(qubit[0]))
    # circuit.add(gates.H(qubit[1]))
    # # circuit.add(gates.H(1))
    # circuit.add(gates.M(*qubit))
    # execution = circuit(nshots=nshots)

    # data.add({"nshots": nshots, "probabilities": [execution.probabilities()]})
    # data.add({"nshots": nshots, "probabilities": [execution.probabilities()]})
    yield data


class UIRS():
    """
    Uniform Independent Random Sequence
    """

    def __init__(self, nqubits, **kwargs):
        """
        """
        self.nqubits = nqubits
        self.gate_generator = None
        self.invert = kwargs.get('invert', False)
        # TODO ask Andrea: what is a Pauli error?
        self.noisemodel = kwargs.get('noisemodel', None)
        # TODO ask Andrea: How does the measuremet gate work and can we also
        # define different ones?
        self.measurement = kwargs.get(
            'measurement', gates.M(*range(nqubits)))

    def __call__(self, sequence_length):
        """ For generating a sequence of circuits the object itself has to be
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
        circuit = models.Circuit(self.nqubits)
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
            circuit.add(circuit.invert().fuse().queue[0])
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
    onequbit_clifford_params = [(0, 0, 0, 0), (np.pi, 1, 0, 0),
        (np.pi,0, 1, 0),
        (np.pi, 0, 0, 1), (np.pi/2, 1, 0, 0), (-np.pi/2, 1, 0, 0),
        (np.pi/2, 0, 1, 0), (-np.pi/2, 0, 1, 0), (np.pi/2, 0, 0, 1),
        (-np.pi/2, 0, 0, 1), (np.pi, 1/np.sqrt(2), 1/np.sqrt(2), 0),
        (np.pi, 1/np.sqrt(2), 0, 1/np.sqrt(2)),
        (np.pi, 0, 1/np.sqrt(2), 1/np.sqrt(2)),
        (np.pi, -1/np.sqrt(2), 1/np.sqrt(2), 0),
        (np.pi, 1/np.sqrt(2), 0, -1/np.sqrt(2)),
        (np.pi, 0, -1/np.sqrt(2), 1/np.sqrt(2)),
        (2*np.pi/3, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)),
        (-2*np.pi/3, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)),
        (2*np.pi/3, -1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)),
        (-2*np.pi/3, -1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)),
        (2*np.pi/3, 1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)),
        (-2*np.pi/3, 1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)),
        (2*np.pi/3, 1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)),
        (-2*np.pi/3, 1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3))]


    def __init__(self, nqubits, **kwargs):
        """
        """
        super().__init__(nqubits, **kwargs)
        # Overwrite the gate generator attribute from the motherclass with
        # the class specific gneerator
        self.gate_generator = self.onequbit_clifford

    
    def Rn(self, theta=0, nx=0, ny=0, nz=0):
        """ 
        """
        matrix = np.array([[np.cos(theta/2) - 1.j*nz*np.sin(theta/2),
                        - ny*np.sin(theta/2) - 1.j*nx*np.sin(theta/2)],
                        [ny*np.sin(theta/2)- 1.j*nx*np.sin(theta/2),
                        np.cos(theta/2) + 1.j*nz*np.sin(theta/2)]])
        return gates.Unitary(matrix, 0)
    
    def onequbit_clifford(self, seed=None):
        """
        """
        if seed is not None:
            backend = get_backend()
            backend.set_seed(seed)

        return self.Rn(*onequbit_clifford_params[
            np.random.randint(len(onequbit_clifford_params))])

class Shadow():
    """ Saves gate-set shadows, e.g. POVM measurement outcome and
    the corresponding gates leading to the state which was measured.

    For convenience the sequence length for the corresponding gates is stored
    along with the used set of POVMs.
    # TODO ask Ingo: What defines an outcome and what POVMs should we use?
    """
    def __init__(self, povms, sequence_lengths, runs, nshots):
        """ Initiate three lists, one for gates, one for the corresponding
        outcomes and one for the corresponding sequence length of the gate.
        """
        self.povms = povms
        self.gate_list = []
        self.outcome_list = []
        self.samples_list = []
        self.probabilities_list = []
        self.sequence_lengths = sequence_lengths
        self.runs = runs
        self.nshots = nshots
    
    def append(self, gate, outcome):
        """ Update the shadow by appending the used gates and the corresponding
        outcomes.
        """
        # Store the gates.
        self.gate_list.append(gate)
        # Convert to list to check shape better.
        outcome = np.array(outcome)
        # Check the outcome type and shape. If it as as many entries as there
        # is shots for the experiment, is must be samples (and if it is a two
        # dimensional array like [[0],[0],...[1]])
        if len(outcome.shape)==2 and outcome.shape[0]==self.nshots:
            # Flatten the outcome list and append it.
            self.samples_list.append(np.array(outcome).flatten())
        elif len(outcome) == 2:
            # This must be the probabilities for 1 qubit either being in
            # ground state or exited state.
            self.probabilities_list.append(np.array(outcome))
        else:
            raise ValueError('The given list doesnt have the right dimension.')
    
    @property
    def samples(self):
        """
        """
        samples = np.array(self.samples_list)
        N = len(self.sequence_lengths)
        samples = samples.reshape((self.runs, N, self.nshots))
        return samples

    @property
    def probabilities(self):
        """
        """
        if self.povms == "Ground State":
            if self.samples_list:
                return 1-np.sum(self.samples, axis=2)\
                    /float(self.samples.shape[-1])
            elif self.probabilities_list:
                return np.array(
                    self.probabilities_list)[:,0].reshape(self.runs, -1) 
    
def experimental_protocol(circuit_generator, myshadow, **kwargs):
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
    # Loop 'runs' many times over the whole protocol.
    for _ in range(runs):
        # Go through every sequence in the protocol.
        for m in sequence_lengths:
            # Generate the right circuits bunched to one gate.
            gate = next(circuit_generator(m),m)
            # Execute the qibo gate.
            executed = gate(nshots=kwargs.get('nshots'))
            # TODO this will be changed in the future.
            try:
                # Get the samples from the executed gate. It should be an
                # object filled with as many integers as used shots.
                outcome = executed.samples()  
            except:
                # Getting the samples is not possible, hence the probabilities
                # have to be stored.
                outcome = executed.probabilities()
                print(outcome)
            # Store the samples.
            myshadow.append(gate, outcome)
            # Store everything.
    return myshadow


def classical_postprocessing(myshadow, **kwargs):
    """ Takes a Shadow object and processes the data inside.
    """
    pass

def exp_func(x,A,f,B):
    """
    """
    return A*f**x+B

def standard_rb_postprocessing(myshadow, **kwargs):
    """ Takes the survival probabilities and fits an exponential curve
    to it.

    Parameters
    ----------
    myshadow : Shadow
    """
    # Extract the probabilities from 'myshadow' dependent on the sequence
    # length m, for each run of the rb protocol there is an array of data.
    pm_runs = myshadow.probabilities
    # They can be averaged or looked at one by one, for now 
    # the average will be used.
    pm = np.sum(pm_runs, axis=0)/myshadow.runs
    m = myshadow.sequence_lengths
    # Calculate an exponential fit to the given data pm dependent on m.
    # 'popt' stores the optimized parameters and pcov the covariance of popt.
    popt, pcov = curve_fit(exp_func, m, pm, p0=[1, 0.98, 0])
    # The variance of the variables in 'popt' are calculated with 'pcov'.
    perr = np.sqrt(np.diag(pcov))
    # Plot the data and the fit.
    plt.plot(m, pm, 'o', label='data')
    x_fit = np.linspace(m[0], m[-1], num=100)
    plt.plot(x_fit, exp_func(x_fit, *popt), '--', label='fit')
    print('A: %f, f: %f, B: %f'%(popt[0], popt[1], popt[2]))


def standard_rb(
    platform,
    qubit : list,
    generator_name,
    sequence_lengths,
    nshots,
):
    # Define the data object.
    # Use the sequence_lengths list as labels for the data
    data = Data("standardrb",
        quantities=list(sequence_lengths))
    # Generate the circuits
    measurement_type = "Ground State"
    runs = 2
    myshadow = Shadow(measurement_type, sequence_lengths, runs, nshots)
    pauli = PauliError(0.05, 0.05, 0.05)
    noise = NoiseModel()
    noise.add(pauli, gates.Unitary)
    if generator_name == 'UIRSOnequbitcliffords':
        mygenerator = UIRSOnequbitcliffords(1, invert=True, noisemodel=False)
    else:
        raise ValueError('This generator is not implemented.')
    myshadow = experimental_protocol(
        mygenerator, myshadow, nshots=nshots)
    for count in range(runs):
        print(myshadow.probabilities)
        data.add({sequence_lengths[i]:myshadow.probabilities[count][i] \
            for i in range(len(sequence_lengths))})
    # pm = np.sum(myshadow.probabilities, axis=0)/runs
    # print(sequence_lengths)
    # print(pm)
    # data.add({'survival_probabilities':pm,
    #     'sequence_lengths':sequence_lengths})
    yield data
