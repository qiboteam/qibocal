TWPA frequency optimization with SNR
====================================

Experiment Description
----------------------

This protocol perform a classification protocol for twpa frequencies in the range [twpa_frequency - frequency_width / 2, twpa_frequency + frequency_width / 2] with step frequency_step.

Example Runcard
---------------

.. code-block::

    - id: twpa
      operation: twpa_frequency_SNR
      parameters:
        frequency_width: 10_000_000
        frequency_step: 1_000_000
        twpa_freq_width: 100_000
        twpa_freq_step: 10_000
        power_level: low


Parameters
----------

- freq_width: width for frequency sweep relative to the readout frequency (Hz)
- freq_step: frequency step for sweep (Hz)
- twpa_freq_width: width for TPWA frequency sweep (Hz)
- twpa_freq_step: TPWA frequency step (Hz)
- power_level: resonator Power regime (low or high)
