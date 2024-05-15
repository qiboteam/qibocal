TWPA frequency optimization
===========================

Experiment Description
----------------------

This protocol perform a classification protocol for twpa frequencies in the range [twpa_frequency - frequency_width / 2, twpa_frequency + frequency_width / 2] with step frequency_step.

Example Runcard
---------------

.. code-block::

    - id: twpa
      operation: twpa_frequency
      parameters:
        frequency_width: 100_000
        frequency_step: 10_000

Parameters
----------

- frequency_width: relative frequency width [Hz]
- frequency_step: frequency step [Hz]
