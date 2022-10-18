from qcvv.data import Data
from qcvv.calibrations.protocols.generators import GeneratorOnequbitcliffords
from qcvv.calibrations.protocols.experiments import Experiment
import pdb
import numpy as np
import pandas as pd

def test_generators():
    """ There are different generators for random circuits from a 
    specific distribution. Here there are tested and shown how to use.
    """
    ############### One qubit Clifford circuits for N qubits ##################
    # 1 qubit case.
    onequbit_circuitgenerator = GeneratorOnequbitcliffords([0])
    # Use the generator (which is actually an iterator)
    # Calling the object it needs a depth/lenght of the circuit, e.g. how many
    # gates there are in the circuit.
    circuit11 = next(onequbit_circuitgenerator(1))
    circuit13 = next(onequbit_circuitgenerator(3))
    # The queue for the circuits should be one and three, respectively.
    assert len(circuit11.queue) == 1
    assert len(circuit13.queue) == 3
    # Calling the circuit generator again should give another circuit (since
    # there are only 24 Clifford gates, it may happen that they are the same).
    # Check if two randomly generated circuits are not the same, three times.
    assert not np.array_equal(
        next(onequbit_circuitgenerator(1)).unitary(),
        next(onequbit_circuitgenerator(1)).unitary()) or not np.array_equal(
        next(onequbit_circuitgenerator(1)).unitary(), 
        next(onequbit_circuitgenerator(1)).unitary())
    # Check the inverse.
    onequbit_circuitgenerator_i = GeneratorOnequbitcliffords([0], invert=True)
    circuit12_i = next(onequbit_circuitgenerator_i(2))
    # The queue should be longer by one now because of the additional
    # inverse gate.
    assert len(circuit12_i.queue) == 3
    # And also the unitary of the circuit should be the identity.
    assert np.allclose(circuit12_i.unitary(), np.eye(2)) 
    # TODO ask Andrea what also can be tested. If the measurement is correct?
    # 5 qubits case.
    # Either give a list of qubits:
    fivequbit_circuitgenerator = GeneratorOnequbitcliffords([0,1,2,3,4])
    # Or just a number, both works.
    fivequbit_circuitgenerator1 = GeneratorOnequbitcliffords(5)
    # The qubit attribute should be the same.
    assert np.array_equal(
        fivequbit_circuitgenerator.qubits, fivequbit_circuitgenerator1.qubits)
    # Generate a radom circuit for 5 qubits with 1 Clifford gate each.
    circuit51 = next(fivequbit_circuitgenerator(1))
    # Generate a radom circuit for 5 qubits with 4 Clifford gate each.
    circuit54 = next(fivequbit_circuitgenerator(4)) 
    assert len(circuit51.queue) == 1
    assert len(circuit54.queue) == 4
    # Check the inverse.
    twoqubit_circuitgenerator_i = GeneratorOnequbitcliffords([0,1], invert=True)
    circuit22_i = next(twoqubit_circuitgenerator_i(2))
    assert np.allclose(circuit22_i.unitary(), np.eye(2**2))
    print('test_generators successfull!')

def test_experiments():
    """ The experiment class has methods to build, save, load and execute
    circuits with a given generator for random circuits. After executing
    the experiment the outcomes are stored.
    """
    
    sequence_lengths = [1,2]
    runs = 3
    qubits = [0]
    mygenerator = GeneratorOnequbitcliffords(qubits)
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    circuits_list = myexperiment.build()
    data_outcome = myexperiment.execute_a_save(nshots=5)

def test_retrieve_experiment():
    """
    """
    # Define the parameters.
    sequence_lengths = [1,2]
    runs = 3
    qubits = [0]
    # Initiate the circuit generator.
    mygenerator = GeneratorOnequbitcliffords(qubits)
    # Initiate the experiment object.
    oldexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    # Build the circuits and save them.
    circuits_list, directory = oldexperiment.build_a_save()
    # Load the circuits and attributes back to a new experiment object.
    newexperiment = Experiment.retreive_experiment(directory)
    # Compare the circuits. They should be the same.
    for countruns in range(runs):
        for countm in range(len(sequence_lengths)): 
            oldcircuit = oldexperiment.circuits_list[countruns][countm]
            newcircuit = newexperiment.circuits_list[countruns][countm]
            assert np.array_equal(oldcircuit.unitary(), newcircuit.unitary())
            assert len(oldcircuit.queue) == len(newcircuit.queue)
    # Also the attributes.
    olddict = oldexperiment.__dict__
    newdict = newexperiment.__dict__
    for key in olddict:
        # The attribute circuits_list was checked above.
        if key != 'circuits_list':
            oldvalue = olddict[key]
            newvalue = newdict[key]
            assert type(oldvalue) == type(newvalue)
            if type(oldvalue) in (str, int, float, bool):
                assert oldvalue == newvalue
            elif type(oldvalue) == list:
                np.array_equal(oldvalue, newvalue)
            else:
                raise TypeError(f'Type {type(oldvalue)} not checked!')
    print('test_retrieve_experiment successfull')

test_generators()
test_retrieve_experiment()

