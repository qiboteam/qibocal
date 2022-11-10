from shutil import rmtree

import numpy as np
from pandas import read_pickle

from qibocal.calibrations.protocol.experiments import Experiment
from qibocal.calibrations.protocols.generators import GeneratorOnequbitcliffords


def test_experiments():
    """ """
    # Just some parameters.
    sequence_lengths = [1, 2]
    runs = 3
    qubits = [0]
    mygenerator = GeneratorOnequbitcliffords(qubits)
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    myexperiment.build()
    myexperiment.execute(nshots=5)
    # Different experiment.
    sequence_lengths = [1, 2]
    runs = 3
    qubits = [0, 1, 2]
    mygenerator = GeneratorOnequbitcliffords(qubits, invert=True)
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    myexperiment.build()
    myexperiment.execute(nshots=5)
    # Again different experiment.
    sequence_lengths = [1, 2]
    runs = 3
    qubits = [0, 1, 2]
    mygenerator = GeneratorOnequbitcliffords(qubits, invert=True)
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs)
    myexperiment.build()
    myexperiment.execute(nshots=10, paulierror_noisparams=[0.01, 0.03, 0.1])


def test_retrieve_from_path():
    """ """
    # Define the parameters.
    sequence_lengths = [1, 2]
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
        if key != "circuits_list":
            oldvalue = olddict[key]
            newvalue = newdict[key]
            assert type(oldvalue) == type(newvalue)
            if type(oldvalue) in (str, int, float, bool):
                assert oldvalue == newvalue
            elif type(oldvalue) == list:
                np.array_equal(oldvalue, newvalue)
            elif issubclass(oldvalue.__class__, Generator):
                assert oldvalue.__class__.__name__ == newvalue.__class__.__name__
            else:
                raise TypeError(f"Type {type(oldvalue)} not checked!")


def test_execute_and_save():
    """ """

    # Set the parameters
    sequence_lengths = [1, 3]
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
    myexperiment = Experiment(mygenerator, sequence_lengths, qubits, runs, nshots=10240)
    myexperiment.build_a_save()
    # Pauli noise paramters.
    noiseparams = [0.1, 0.1, 0.05]
    # Inject the noise while executing.
    myexperiment.execute_a_save(paulierror_noiseparams=noiseparams)
    directory2 = myexperiment.directory
    # Load into new object.
    recexperiment = Experiment.retrieve_from_path(directory2)
    # There is a new attribute!
    assert not hasattr(myexperiment, "paulierror_noiseparams") and hasattr(
        recexperiment, "paulierror_noiseparams"
    )
    assert recexperiment.paulierror_noiseparams == noiseparams
    # Also, the natural probabilities should differ from the one calculated
    # with the samples!
    recprobs = recexperiment.probabilities(from_samples=False)
    recprobs_fromsamples = recexperiment.probabilities(from_samples=True)
    assert recprobs.shape == recprobs_fromsamples.shape
    assert not np.allclose(recprobs, recprobs_fromsamples, atol=1e-1)
    rmtree(directory2)


def test_retrieve_from_dataobjects():
    """ """
    # Set the parameters
    sequence_lengths = [1, 2, 5]
    runs = 2
    # Set two qubits
    qubits = [0, 1]
    nshots = 10
    # Initiate the circuit generator abd the experiment.
    mygenerator = GeneratorOnequbitcliffords(qubits)
    oldexperiment = Experiment(
        mygenerator, sequence_lengths, qubits, runs, nshots=nshots
    )
    # Build the cirucuits, the yare stored as attribute in the object.
    oldexperiment.build_a_save()
    # Execute the experiment, e.g. the single circuits, store the outcomes
    # as attributes and in a file.
    oldexperiment.execute_a_save()
    directory = oldexperiment.directory
    data_samples = Data("samples", quantities=list(sequence_lengths))
    data_samples.df = read_pickle(f"{directory}samples.pkl")
    data_probs = Data("probabilities", quantities=list(sequence_lengths))
    data_probs.df = read_pickle(f"{directory}probabilities.pkl")
    data_circs = Data("circuits", quantities=list(sequence_lengths))

    data_circs.df = read_pickle(f"{directory}circuits.pkl")
    recexperiment = Experiment.retrieve_from_dataobjects(
        data_circs, data_samples, data_probs
    )
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
        if key not in ("circuits_list", "inverse", "directory"):
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
                raise TypeError(f"Type {type(oldvalue)} not checked!")
    rmtree(directory)


def test_filter_single_qubit():
    """ """
    # Define the parameters.
    sequence_lengths = [1, 2, 5]
    runs = 2
    nshots = 5
    qubits = [0]
    # Initiate the circuit generator.
    mygenerator = GeneratorOnequbitcliffords(qubits, invert=False)
    # Initiate the experiment object.
    experiment = Experiment(mygenerator, sequence_lengths, qubits, runs, nshots)
    # Build the circuits.
    experiment.build()
    # Execute the experiment.
    experiment.execute_experiment(paulierror_noiseparams=[0.1, 0.1, 0.1])
    # Get the filter array.
    filters = np.average(experiment.filter_single_qubit(), axis=0)
    experiment.plot_scatterruns(use_probs=True)
