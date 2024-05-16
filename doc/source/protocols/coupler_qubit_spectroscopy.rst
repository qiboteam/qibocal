Coupler qubit spectroscopy
==========================

Experiment Description
----------------------

This consist on a frequency sweep on the qubit frequency while we change the flux coupler pulse amplitude of the coupler pulse.
We expect to enable the coupler during the amplitude sweep and detect an avoided crossing that will be followed by the frequency sweep.
This needs the qubits at resonance, the routine assumes a sweetspot value for the higher frequency qubit that moves it to the lower frequency qubit instead of trying to calibrate both pulses at once.
This should be run after qubit_spectroscopy to further adjust the coupler sweetspot if needed and get some information on the flux coupler pulse amplitude requiered to enable 2q interactions.

Example Runcard
---------------

.. code-block::

    - id: coupler qubit spectroscopy
      operation: coupler_qubit_spectroscopy
      parameters:
        measured_qubits: 0
        amplitude: 0.1
        drive_duration: 200

Parameters
----------

- measured_qubits: qubit to measure from the pair
- amplitude: readout or qubit drive amplitude (optional). If defined, same amplitude will be used in all qubits. Otherwise the default amplitude defined on the platform runcard will be used
- drive_duration: drive pulse duration to excite the qubit before the measuremen
