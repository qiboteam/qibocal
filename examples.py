from qcvv.data import Data
from qcvv.calibrations.protocols.generators import *
from qcvv.calibrations.protocols.experiments import Experiment
from shutil import rmtree
from pandas import read_pickle
import pdb
import numpy as np

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
    # Try the act_on variabl.
    fourqubit_circuitgenerator = GeneratorOnequbitcliffords(
        [0,1,2,3], act_on=2, invert=True)
    circuit41 = next(fourqubit_circuitgenerator(2))
    print('This circuit should only act on the third qubit!')
    print(circuit41.draw())
    print('test_generators successfull!')

def test_experiments():
    """
    """
    # Just some parameters.
    sequence_lengths = [1,2]
    runs = 3
    qubits = [0]
    mygenerator = GeneratorOnequbitcliffords(qubits)
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    myexperiment.build()
    myexperiment.execute(nshots=5)
    # Different experiment.
    sequence_lengths = [1,2]
    runs = 3
    qubits = [0,1,2]
    mygenerator = GeneratorOnequbitcliffords(qubits, invert=True)
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    myexperiment.build()
    myexperiment.execute(nshots=5)
    # Again different experiment.
    sequence_lengths = [1,2]
    runs = 3
    qubits = [0,1,2]
    mygenerator = GeneratorOnequbitcliffords(qubits, invert=True)
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    myexperiment.build()
    myexperiment.execute(nshots=10, paulierror_noisparams=[0.01,0.03,0.1])
    print('test_experiment successfull!')

def test_retrieve_from_path():
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
    oldexperiment.build_a_save()
    # Get the directory.
    directory = oldexperiment.directory
    # Load the circuits and attributes back to a new experiment object.
    recexperiment = Experiment.retrieve_from_path(directory)
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
            elif issubclass(oldvalue.__class__, Generator):
                assert oldvalue.__class__.__name__ \
                    ==  newvalue.__class__.__name__
            else:
                raise TypeError(f'Type {type(oldvalue)} not checked!')
    print('test_retrieve_experiment successfull')

def test_execute_and_save():
    """
    """

    # Set the parameters
    sequence_lengths = [1,3]
    runs = 2
    qubits = [0]
    mygenerator = GeneratorOnequbitcliffords(qubits)
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    # Build the circuits and save them.
    myexperiment.build_a_save()
    # Execute the experiment and save the outcome.
    myexperiment.execute_a_save()
    probs = myexperiment.outcome_probabilities
    # Get the directory where the circuits were stored.
    directory1 = myexperiment.directory
    # Load the experiment from the files.
    recexperiment = Experiment.retrieve_from_path(directory1)
    # Also, load the outcome.
    recexperiment.load_probabilities(directory1)
    recprobs = recexperiment.outcome_probabilities
    for count_runs in range(runs):
        assert np.array_equal(probs[count_runs], recprobs[count_runs])
    # Remove the directory.
    rmtree(directory1)
    # Make a new experiment, this time with some injected noise!
    mygenerator = GeneratorOnequbitcliffords(qubits, invert=True)
    myexperiment = Experiment(
        mygenerator, sequence_lengths, qubits, runs, nshots=10240)
    myexperiment.build_a_save()
    # Pauli noise paramters.
    noiseparams = [0.1, 0.1, 0.05]
    # Inject the noise while executing.
    myexperiment.execute_a_save(paulierror_noiseparams=noiseparams)
    directory2 = myexperiment.directory
    # Load into new object.
    recexperiment = Experiment.retrieve_from_path(directory2)
    # There is a new attribute!
    assert not hasattr(myexperiment, 'paulierror_noiseparams') \
        and hasattr(recexperiment, 'paulierror_noiseparams')
    assert recexperiment.paulierror_noiseparams == noiseparams
    # Also, the natural probabilities should differ from the one calculated
    # with the samples!
    recprobs = recexperiment.probabilities(from_samples=False)
    recprobs_fromsamples = recexperiment.probabilities(from_samples=True)
    assert recprobs.shape == recprobs_fromsamples.shape
    assert not np.allclose(recprobs, recprobs_fromsamples, atol=1e-1)
    rmtree(directory2)
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
    # Get the probabilities return from the executed circuit itself,
    # which is the same as myexperiment.probabilities(from_samples=False).
    probs_natural = np.average(myexperiment.outcome_probabilities, axis=0)
    # Check if they are close.
    assert np.allclose(probs_fromsamples, probs_natural, atol=1e-03)
    # For fitting and error bar estimation the probabilities of every run
    # are needed, meaning that there is no average over the runs.
    probs_runs = myexperiment.probabilities(averaged=False)
    assert probs_runs.shape == (runs, len(sequence_lengths), 2**len(qubits))
    print('test_probabilities_a_samples successfull!')

def test_retrieve_from_dataobjects():
    """
    """
     # Set the parameters
    sequence_lengths = [1, 2, 5]
    runs = 2
    # Set two qubits
    qubits = [0, 1]
    nshots = None
    # Initiate the circuit generator abd the experiment.
    mygenerator = GeneratorOnequbitcliffords(qubits)
    oldexperiment = Experiment(
        mygenerator, sequence_lengths, qubits, runs, nshots=nshots)
    # Build the cirucuits, the yare stored as attribute in the object.
    oldexperiment.build_a_save()
    # Execute the experiment, e.g. the single circuits, store the outcomes
    # as attributes and in a file.
    oldexperiment.execute_a_save()
    directory = oldexperiment.directory
    data_samples = Data(
            'samples', quantities=list(sequence_lengths))
    data_samples.df = read_pickle(f'{directory}samples.pkl')
    data_probs = Data(
            'probabilities', quantities=list(sequence_lengths))
    data_probs.df = read_pickle(f'{directory}probabilities.pkl')
    data_circs = Data(
            'circuits', quantities=list(sequence_lengths))
    data_circs.df = read_pickle(f'{directory}circuits.pkl')
    recexperiment = Experiment.retrieve_from_dataobjects(
        data_circs, data_samples, data_probs)
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
        if key not in ('circuits_list', 'inverse', 'directory'):
            oldvalue = olddict[key]
            newvalue = newdict[key]
            if type(oldvalue) in (str, int, float, bool):
                assert oldvalue == newvalue
            elif type(oldvalue) == list:
                np.array_equal(oldvalue, newvalue)
            elif issubclass(oldvalue.__class__, Generator):
                # Did not figure out yet how to retrieve the random
                # circuit generator.
                pass
            elif oldvalue is None:
                assert oldvalue is oldvalue
            else:
                raise TypeError(f'Type {type(oldvalue)} not checked!')
    print('test_retrieve_experiment successfull')
    rmtree(directory)


# test_generators()
# test_retrieve_from_path()
# test_execute_and_save()
# test_probabilities_a_samples()
test_retrieve_from_dataobjects()
