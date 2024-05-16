CZ virtual-z signal calibration
===============================

Experiment Description
----------------------

Check the two-qubit landscape created by a flux pulse of a given duration and amplitude.
The system is initialized with a Y90 pulse on the low frequency qubit and either an Id or an X gate on the high frequency qubit. Then the flux pulse is applied to the high frequency qubit in order to perform a two-qubit interaction. The Id/X gate is undone in the high frequency qubit and a theta90 pulse is applied to the low frequency qubit before measurement. That is, a pi-half pulse around the relative phase parametereized by the angle theta.
Measurements on the low frequency qubit yield the 2Q-phase of the gate and the remnant single qubit Z phase aquired during the execution to be corrected.
Population of the high frequency qubit yield the leakage to the non-computational states during the execution of the flux pulse.

This experiemnt uses arbitrary V-like units.

Example Runcard
---------------

.. code-block::

    - id: cz virtualz
      operation: cz_virtualz
      parameters:
        theta_start: 0
        theta_end: 10
        theta_step: 1
        flux_pulse_amplitude: 0.3
        flux_pulse_duration: 10
        dt: 8
        parking: False

Parameters
----------

- theta_start: initial angle for the low frequency qubit measurement in radians
- theta_end: final angle for the low frequency qubit measurement in radians
- theta_step: step size for the theta sweep in radians
- flux_pulse_amplitude: amplitude of flux pulse implementing CZ
- flux_pulse_duration: duration of flux pulse implementing CZ
- dt: time delay between flux pulses and readout
- parking: wether to park non interacting qubits or not
