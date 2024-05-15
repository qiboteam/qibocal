Readout optimization: resonator frequency
=========================================

Experiment Description
----------------------

Fine tuning optimization of the readout pulse.
This protocol sweeps the readout frequency performing a classification routine and evaluating the error probability at each step.
The sweep will be interrupted if the probability error is less than the `error_threshold`.

Example Runcard
---------------

.. code-block::

    - id: readout_mitigation_matrix
      operation: readout_mitigation_matrix
      parameters:
        nshots: 10_000
        pulses: True

Parameters
----------

- frequency_step: frequency step to be probed
- frequency_start: frequency_start to be probed
- frequency_stop: frequency stop value
- error_threshold: probability error threshold to stop the best frequency search
