.. _runcard:

How to execute calibration protocols in ``Qibocal``?
====================================================

In ``Qibocal`` we adopt a declarative programming paradigm, i.e. the user should specify directly
what he wants to do without caring about the underlying implementation.

This paradigm is implemented in ``Qibocal`` in the form of runcards. A runcard is essentially
a set of instructions that are specified in a file.

Down below we present how to write a runcard to execute calibration protocols using `qq`.

.. code-block:: yaml

    backend: <qibo backend>

    platform: <qibolab platform name>

    targets: <list of targets where all protocols will be executed.>


    actions:
        - id: <first protocol id>
          operation: <first qibocal protocol>
          parameters:
            arg1: ...
            arg2: ...

        - id: <second protocol id>
          operation: <second qibocal protocol>
          parameters:
            arg1: ...
            arg2: ...

Here is a description of the global parameters to be specified:
    * ``backend``: ``Qibo`` backend, if not provided ``Qibolab`` will be used.
    * ``platform``: QPU where the experiments will be executed. Possible choices
        for TII users are available in this `repository <https://github.com/qiboteam/qibolab_platforms_qrc>`_.
        For non-TII users it is possible a setup a custom platform using  `Qibolab <https://qibo.science/qibolab/stable/tutorials/lab.html>`_.
    * ``targets``: list of ``QubitId`` for a specific platform. It can also be a list of ``QubitPairId``
        in the case of protocols for qubit pairs, or a ``list[list[QubitId]]`` for protocols targeting groups of qubits.

Under ``actions`` are listed the protocols that will be executed sequentially.

For each protocol the user needs to specify the following:
    * ``id``: custom id chosen by the user.
    * ``operation``: protocol available in ``Qibocal``. See :ref:`calibration_routines` for a complete list of the protocols available.
    * ``parameters``: input parameters.
