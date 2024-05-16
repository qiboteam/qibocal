TWPA frequency-power optimization
=================================

Experiment Description
----------------------

This protocol perform a classification protocol for twpa frequencies in the range [twpa_frequency - frequency_width / 2, twpa_frequency + frequency_width / 2] with step frequency_step and powers in the range [twpa_power - power_width / 2, twpa_power + power_width / 2]

Example Runcard
---------------

.. code-block::

    - id: twpa
      operation: twpa_frequency_power
      parameters:
        frequency_width: 10_000_000
        frequency_step: 1_000_000
        power_width: 10
        power_step: 1

Parameters
----------

- frequency_width: frequency total width
- frequency_step: frequency step to be probed
- power_width: power total width
- power_step: power step to be probed
