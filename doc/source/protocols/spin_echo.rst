Spin-Echo
=========

Experiment Description
----------------------

The Spin-Echo experiment enables to measure the intrinsic T2 of the qubit.
It is composed of two pi-half pulses with a delay in between.
Differently from the Ramsey experiment, in the middle of the delay there is also a pi-pulse.

This routine plots and fits probabilities.

Example Runcard
---------------

.. code-block::

    - id: spin_echo
      operation: spin_echo
      parameters:
        delay_between_pulses_start: 4
        delay_between_pulses_end: 10000
        delay_between_pulses_step: 200
        unrolling: False

Parameters
----------

- delay_between_pulses_start: start value for the delay
- delay_between_pulses_end: end value for the delay
- delay_between_pulses_step: step value for the delay
- unrolling: if True, it uses sequence unrolling to deploy the sequences faster.
