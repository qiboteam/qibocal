Rabi-length experiment
======================

Experiment Description
----------------------

Data acquisition for Rabi experiment sweeping amplitude.
In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length to find the drive pulse length that creates a rotation of a desired angle.

This routine uses probabilities.

Example Runcard
---------------

.. code-block::

    - id: rabi length
      operation: rabi_length
      parameters:
        pulse_duration_start: 4
        pulse_duration_end: 84
        pulse_duration_step: 8
        pulse_amplitude: 0.5
        relaxation_time: 100_000
        nshots: 1024

Parameters
----------

- pulse_duration_start: minimum pi-pulse length
- pulse_duration_end: maximum pi-pulse length
- pulse_duration_step: step of the length of the pi-pulse
- pulse_amplitude: pi-pulse amplitude
