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
    recexperiment = Experiment.retrieve_experiment(directory)
    # Compare the circuits. They should be the same.
    for countruns in range(runs):
        for countm in range(len(sequence_lengths)): 
            circuit = oldexperiment.circuits_list[countruns][countm]
            reccircuit = recexperiment.circuits_list[countruns][countm]
            assert np.array_equal(circuit.unitary(), reccircuit.unitary())
            assert len(circuit.queue) == len(reccircuit.queue)
    # Also the attributes.
    olddict = oldexperiment.__dict__
    newdict = recexperiment.__dict__
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

def test_execute_and_save():
    """
    """
    from shutil import rmtree
    from ast import literal_eval
    # Set the parameters
    sequence_lengths = [1,2,5,10]
    runs = 3
    qubits = [0]
    mygenerator = GeneratorOnequbitcliffords(qubits)
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    # Build the circuits and save them.
    circuits_list, directory = myexperiment.build_a_save()
    # Execute the experiment and save the outcome.
    samples, probs = myexperiment.execute_a_save()
    # Load the experiment from the files.
    recexperiment = Experiment.retrieve_experiment(directory)
    # Also, load the outcome.
    recprobs = recexperiment.load_outcome(directory)
    for count_runs in range(runs):
        assert np.array_equal(probs[count_runs], recprobs[count_runs])
    # Remove the directory.
    rmtree(directory)
    # Make a new experiment, this time with some injected noise!
    mygenerator = GeneratorOnequbitcliffords(qubits, invert=True)
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    myexperiment.build()
    samples, probs = myexperiment.execute_a_save(
        paulierror_noiseparams=[0.1,0.1,0.1])
    samples = np.array(myexperiment.outcome_samples)
    probabilities = np.average(
        samples, axis=1).reshape(runs, len(sequence_lengths))
    pdb.set_trace()
    print(np.array(probs))
    rmtree(myexperiment.directory)
    print('test_execute_and_save successfull!')

def test_probabilities_a_samples():
    """
    """
    # Set the parameters
    sequence_lengths = [1, 2, 5]
    runs = 2
    # Set two qubits
    qubits = [0, 1]
    # Put the shots up a lot to see if the samples yield the same probabilities.
    nshots = int(1e6)
    # Initiate the circuit generator abd the experiment.
    mygenerator = GeneratorOnequbitcliffords(qubits)
    myexperiment = Experiment(
        mygenerator, sequence_lengths, qubits, runs, nshots=nshots)
    # Build the cirucuits, the yare stored as attribute in the object.
    myexperiment.build()
    # Execute the experiment, e.g. the single circuits, store the outcomes
    # as attributes.
    myexperiment.execute_experiment()
    assert myexperiment.samples().shape == (
        runs, len(sequence_lengths), nshots, len(qubits))
    # Get the probabilities calculated with the outcome samples.
    probs_fromsamples = myexperiment.probabilities(averaged=True)
    # Get the probabilities return from the executed circuit itself.
    probs_natural = np.average(myexperiment.outcome_probs, axis=0)
    # Check if they are close.
    assert np.allclose(probs_fromsamples, probs_natural, atol=1e-03)
    # For fitting and error bar estimation the probabilities of every run
    # are needed, meaning that there is no average over the runs.
    probs_runs = myexperiment.probabilities(averaged=False)
    assert probs_runs.shape == (runs, len(sequence_lengths), 2**len(qubits))
    print('test_probabilities_a_samples successfull!')




# test_generators()
# test_retrieve_experiment()
# test_execute_and_save()
test_probabilities_a_samples()

