Qubit spectroscopy
==================

Experiment Description
----------------------

Two-tone spectroscopy experiment. First a drive pulse is fired, with a variable frequency, then a measurement is performed.

The objective is to find the qubit resonator frequency, for the transition 0 -> 1.

Example Runcard
---------------

.. code-block::

    - id: qubit spectroscopy
      operation: qubit_spectroscopy
      parameters:
        drive_amplitude: 0.005
        drive_duration: 2000
        freq_width: 10_000_000
        freq_step: 100_000
        nshots: 1024
        relaxation_time: 0

Parameters
----------

- freq_width: width [Hz] for frequency sweep relative  to the qubit frequency
- freq_step: frequency [Hz] step for sweep
- drive_duration: drive pulse duration [ns]. Same for all qubits
- drive_amplitude: optional amplitude for all pulses
