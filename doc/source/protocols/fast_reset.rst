Fast reset experiment
=====================

Experiment Description
----------------------

Test fast reset parameters, performing measurements followed by pi-pulse in the case of a \|1> state.

Example Runcard
---------------

.. code-block::

    - id: fast_reset
      operation: fast_reset
      parameters:
        nshots: 1024
