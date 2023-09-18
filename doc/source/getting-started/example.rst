.. _example:

Minimal working example
=======================

This section shows the steps to perform a resonator spectroscopy with Qibocal.

To run a resonator spectroscopy, following the instructions
presented in :ref:`runcard`, we can use this runcard.

.. code-block:: yaml

    platform: qw5q_gold_qblox

    qubits: [0]

    actions:
        - id: resonator high power high amplitude
        priority: 0
        operation: resonator_spectroscopy
        parameters:
            freq_width: 10_000_000
            freq_step: 100_000
            amplitude: 0.4
            power_level: high
            nshots: 1024

More examples of runcards are available on `Github <https://github.com/qiboteam/qibocal/tree/main/runcards>`_ .

Run the routine
^^^^^^^^^^^^^^^

.. code-block::

    qq auto example.yml -o resonator_spectroscopy_routine
