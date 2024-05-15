Rabi-amplitude-signal experiment
================================

Experiment Description
----------------------

Data acquisition for Rabi experiment sweeping amplitude.
In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude to find the drive pulse amplitude that creates a rotation of a desired angle.

This routine uses arbitrary V-like units.

Example Runcard
---------------

.. code-block::

    - id: rabi
      operation: rabi_amplitude_signal
      parameters:
        min_amp_factor: 0.0
        max_amp_factor: 2.0
        step_amp_factor: 0.02
        pulse_length: 40
        relaxation_time: 100_000
        nshots: 1024

Parameters
----------

- min_amp_factor: minimum multiplicative factor for the pi-pulse amplitude
- max_amp_factor: minimum multiplicative factor for the pi-pulse amplitude
- step_amp_factor: minimum multiplicative factor for the pi-pulse amplitude
- pulse_length: pi-pulse duration
