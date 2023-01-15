.. _runcard:

How to write your runcard?
==========================

In ``qibocal`` we adopt a declarative programming paradigm, i.e. the user should specify directly
what he wants to do without caring about the underlying implementation.

This paradigm is implemented in ``qibocal`` in the form of runcards. A runcard will contain all
the essential information to run a specific task.

In the case of the ``qq`` command a possible runcard should look like this:

.. code-block:: yaml

    platform: tii5q

    runcard: <path_to_platform_runcard>

    qubits: [0]

    format: pickle

    actions:
        first_routine:
            arg1: ...
            arg2: ...

        second_routine:
            arg1: ...
            arg2: ...


First, the user will need to specify some global parameters including:
    * ``platform``: the platform name.
    * ``runcard``: path to the platform runcard (optional). If not specified it will be used the platform runcard available in qibolab.
    * ``qubits``: the qubit(s) that we are calibrating.
    * ``format``: the format for storing the measurements.

After those the user will simply populate ``actions`` with all the routines
that he will like to run followed by their respective arguments.
