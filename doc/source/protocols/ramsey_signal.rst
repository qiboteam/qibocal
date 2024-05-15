T2 Ramsey signal (detuned)
==========================

Experiment Description
----------------------

The T2 experiment enables to measure T2* of the qubit.
It is composed of two pi-half pulses with a delay in between.

This routine uses arbitrary unit (V-like).

Differently from T2, the routine enable to define a fake detuning, helping to find the correct drive frequency, but making it harder to fit.

Example Runcard
---------------

.. code-block::

    - id: t2
      operation: t2_signal
      parameters:
        delay_between_pulses_start: 16
        delay_between_pulses_end: 20000
        delay_between_pulses_step: 100
        detuning: 10_000_000
        nshots: 10

Parameters
----------

- delay_between_pulses_start: start value for the delay
- delay_between_pulses_end: end value for the delay
- delay_between_pulses_step: step value for the delay
- detuning: frequency detuning [Hz]
- single_shot: if True saves single-shot data
