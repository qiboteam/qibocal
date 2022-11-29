
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

.. code-block:: python
    # Define the necessary variables.
    nqubits = 1 # How many circuits in the quantum hardware.
    depths = [0,1,5,10] # How many random gates there are in each circuit.
    runs = 2 # The amount
    # 
