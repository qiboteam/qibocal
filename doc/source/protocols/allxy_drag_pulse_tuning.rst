AllXY_drag_pulse_tuning
=======================

Experiment Description
----------------------

The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the \|0> state) is subjected to two back-to-back single-qubit gates and measured.

In each round, we run 21 different gate pairs: ideally, the first 5 return the qubit to \|0>, the next 12 drive it to superposition state, and the last 4 put the qubit in \|1> state.

The AllXY iteration method allows the user to execute iteratively the list of gates playing with the drag pulse shape in order to find the optimal drag pulse coefficient for pi pulses.

Example Runcard
---------------

.. code-block::

    - id: drag_pulse_tuning
      operation: allxy_drag_pulse_tuning
      parameters:
        beta_start: -0.2
        beta_end: 0.2
        beta_step: 0.1

Parameters
----------

- beta_start: start value for the beta parameter (for Drag pulses)
- beta_end: end value for the beta parameter (for Drag pulses)
- beta_step: step value for the beta parameter (for Drag pulses)
