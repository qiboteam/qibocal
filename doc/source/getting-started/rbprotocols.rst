
Randomized Benchmarking Protocols
=================================

``qibocal`` provides a convenient infrastructure to implement randomized benchmarking protocols
easily and fast. In ``abstract.py`` the overall structure is set up.
The foundation is three classes:
    1. The circuit factory (INSERT LINK), an iterator which produces circuits when called.
    2. The Experiment class (INSERT LINK) which takes an iterable object producing circuits, optional some parameters. It is able to execute the circuits and to overwrite/store/process the necessary data.
    3. A class (INSERT LINK) storing and displaying the results of a randomized benchmarking scheme.

Standard Randomized Benchmarking 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First define the necessary variables which will be used when initiating the 
circuit factory and the experiment object.

.. code-block:: python

    # Define the necessary variables.
    nqubits = 1 # Number of qubits in the quantum hardware.
    depths = [0,1,4] # How many random gates there are in each circuit.
    runs = 2 # The amount of repetitions of the whole experiment.
    nshots = 5 # When a circuit is executed how many shots are used.

The circuit factory
"""""""""""""""""""

Now build the circuit factory, and check out how it works.

.. code-block:: python

    from qibocal.calibrations.protocols import standardrb
    # To not alter the iterator when using it, make deep copies.
    from copy import deepcopy
    factory = standardrb.SingleCliffordsInvFactory(nqubits, depths, runs)
    # ``factory`` is an iterator class object generating single clifford
    # gates with the last gate always the inverse of the whole gate sequence.
    # There are mainly three ways how to extract the circuits.
    # 1. Make a list out of the iterator object.
    circuits_list1 = list(deepcopy(factory))
    # 2. Use a for loop.
    circuits_list2 = []
    for circuit in deepcopy(factory):
        circuits_list2.append(circuit)
    # 3. Make an iterator and extract the circuits with the next method.
    iter_factory = iter(deepcopy(factory))
    circuits_list3, iterate = [], True
    while iterate:
        try:
            circuits_list3.append(next(iter_factory))
        except StopIteration:
            iterate = False
    # All the three lists have circuits constructed with
    # single clifford gates according to the ``depths``list,
    # repeated ``runs``many times.

The experiment
""""""""""""""

.. code-block:: python

    # Initiate the standard RB experiment. To make it simpler
    # first without simulated noise on the circuits. 
    experiment = standardrb.StandardRBExperiment(factory, nshots)
    # Nothing happened yet. The experiment has to be executed
    # to execute the single circuits and store the samples along
    # with the number of applied gates.
    experiment.execute()
    # Check out the data in a data frame. Since there is no noise all
    # the samples from the measured qubits were in the ground state.
    print(experiment.dataframe)
    #                     samples  depth
    # 0  [[0], [0], [0], [0], [0]]      0
    # 1  [[0], [0], [0], [0], [0]]      1
    # 2  [[0], [0], [0], [0], [0]]      5
    # 3  [[0], [0], [0], [0], [0]]      0
    # 4  [[0], [0], [0], [0], [0]]      1
    # 5  [[0], [0], [0], [0], [0]]      5

The postprocessing
""""""""""""""""""

The standard randomized benchmarking protocol aims at analyzing the probability
of the state coming back to the inital state when inversing all the gates applied gates.
Since normally the initial state is the grounds state :math:`\ket{0}` the survival 
of the ground state probability has to be analyzed.
And with analyzed it is meant to extract the probabilities for every sequence (or depth)
of each run, average over the runs, fit an exponential decay to the signal and use the
base of the exponent to calculate the average gate fidelity.

.. code-block:: python

    from qibocal.calibrations.protocols.fitting_methods import fit_exp1_func
    # Make the experiment calculate its own ground state probability,
    # it will be appended to the data.
    experiment.apply(standardrb.groundstate_probability)
    # Now the data attribute of the experiment object has all its needs
    # for the desired signal (ground state survival probability) to
    # be fitted and plotted.
    # For that use the custom designed ``Result``class, use a single
    # exponential decay model for fitting.
    result = standardrb.StandardRBResult(experiment.dataframe, fit_exp1_func)
    # With the result class multiple figure can be build and stored and when
    # the report is needed all of these figure will be shown in one report.
    result.single_fig()
    result.report().show()