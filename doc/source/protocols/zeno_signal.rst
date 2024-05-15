Zeno-signal experiment
======================

Experiment Description
----------------------

In a T1_Zeno experiment, we measure an excited qubit repeatedly. Due to decoherence processes, it is possible that, at the time of measurement, the qubit will not be excited anymore.
The quantum zeno effect consists of measuring allowing a particle's time evolution to be slowed down by measuring it frequently enough. However, in the experiments we see that due the QND-ness of the readout pulse that the qubit decoheres faster.
    Reference: https://link.aps.org/accepted/10.1103/PhysRevLett.118.240401.

This routine does not use probabilities, but aribtrary units.

Example Runcard
---------------

.. code-block::

    - id: zeno
      operation: zeno
      parameters:
        readouts: 20

Parameters
----------

- readouts: number of readout pulses
