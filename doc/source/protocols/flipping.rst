Flipping
========

Experiment Description
----------------------

The flipping experiment correct the delta amplitude in the qubit drive pulse. We measure a qubit after applying a Rx(pi/2) and N flips (Rx(pi) rotations). After fitting we can obtain the delta amplitude to refine pi pulses.

This experiment uses probabilities.

Example Runcard
---------------

.. code-block::

    - id: flipping
      operation: flipping
      parameters:
        nflips_max: 20
        nflips_step: 1

Parameters
----------

- nflips_max: maximum number of flips (couple of pi-pulses)
- nflips_step: step-number of flips (couple of pi-pulses)
