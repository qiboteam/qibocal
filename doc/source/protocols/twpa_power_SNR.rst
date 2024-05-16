TWPA power optimization with SNR
================================

Experiment Description
----------------------

This protocol perform a classification protocol for twpa powers in the range [twpa_power - power_width / 2, twpa_power + power_width / 2] with step power_step.

Example Runcard
---------------

.. code-block::

    - id: twpa
      operation: twpa_power_SNR
      parameters:
        power_width: 10
        power_step: 1

Parameters
----------

- power_width: power total width
- power_step: power step to be probed
