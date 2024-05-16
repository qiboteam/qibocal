Chevron (two-qubit interaction)
===============================

Experiment Description
----------------------

Perform an CZ experiment between pairs of qubits by changing its frequency.

This experiment uses probabilities.

Example Runcard
---------------

.. code-block::

    - id: allXY
      operation: allxy
      parameters:
        amplitude_min_factor: 0
        amplitude_max_factor: 1
        amplitude_step_factor: 0.1
        duration_min: 10
        duration_max: 1000
        duration_step: 20
        dt: 8
        parking: False

Parameters
----------

- amplitude_min_factor: amplitude minimum
- amplitude_max_factor: amplitude maximum
- amplitude_step_factor: amplitude step
- duration_min: duration minimum
- duration_max: duration maximum
- duration_step: duration step
- dt: time delay between flux pulses and readout
- parking: wether to park non interacting qubits or not
