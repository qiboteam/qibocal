T2 Ramsey signal
================

Experiment Description
----------------------

The T2 experiment enables to measure T2* of the qubit.
It is composed of two pi-half pulses with a delay in between.

This routine uses arbitrary unit (V-like).

Example Runcard
---------------

.. code-block::

    - id: t2
      operation: t2_signal
      parameters:
        delay_between_pulses_start: 16
        delay_between_pulses_end: 20000
        delay_between_pulses_step: 100
        nshots: 10

Parameters
----------

- delay_between_pulses_start: start value for the delay
- delay_between_pulses_end: end value for the delay
- delay_between_pulses_step: step value for the delay
- single_shot: if True saves single-shot data
