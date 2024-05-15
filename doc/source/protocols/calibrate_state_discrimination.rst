Calibrate state-discrimination
==============================

Experiment Description
----------------------

Calculates the optimal kernel for the readout. It has to be run one qubit at a time.
The kernels are stored in the result.npz generated on the report.

Example Runcard
---------------

.. code-block::

    - id: calibrate states
      operation: calibrate_state_discrimination
      parameters:
        shots: 1000
