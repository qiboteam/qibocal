Readout mitigation matrix
=========================

Experiment Description
----------------------

Measure the rates of error between prepared and measured states.

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

- pulses: if True, get readout mitigation matrix using pulses. If False gates will be used
