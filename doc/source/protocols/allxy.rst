AllXY
=====

Experiment Description
----------------------

The AllXY experiment is a simple test of the calibration of single qubit gates.
The qubit, initialized in the \|0⟩ state, is subjected to two back-to-back single-qubit gates and then measured.

In each round, we run 21 different gate pairs: ideally, the first 5 return the qubit to the \|0⟩ state, the next 12 drive it to a superposition state, and the last 4 put the qubit in the \|1⟩ state.

Example Runcard
---------------

.. code-block::

    - id: allXY
      operation: allxy
      parameters:
        beta_param: null
        unrolling: False

Parameters
----------

- beta_param: Beta parameter for drag pulses.
- unrolling: if True, it uses sequence unrolling to deploy the sequences faster.
