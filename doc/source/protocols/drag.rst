Drag tuning
===========

Experiment Description
----------------------

See https://arxiv.org/pdf/1504.06597.pdf Fig. 2 (c).

Example Runcard
---------------

.. code-block::

    - id: drag_pulse_tuning
      operation: drag_pulse_tuning
      parameters:
        beta_start: 0
        beta_end: 0.02
        beta_step: 0.01
        nshots: 10

Parameters
----------

- beta_start: DRAG pulse beta start sweep parameter
- beta_end: DRAG pulse beta end sweep parameter
- beta_step: DRAG pulse beta sweep step parameter
- unrolling: if True unrolls the sequences for a faster execution
