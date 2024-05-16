Coupler resonator spectroscopy
==============================

Experiment Description
----------------------

This consist on a frequency sweep on the readout frequency while we change the flux coupler pulse amplitude of the coupler pulse.
We expect to enable the coupler during the amplitude sweep and detect an avoided crossing that will be followed by the frequency sweep.
No need to have the qubits at resonance. This should be run after resonator_spectroscopy to detect couplers and adjust the coupler sweetspot if needed and get some information on the flux coupler pulse amplitude requiered to enable 2q interactions.

Example Runcard
---------------

.. code-block::

    - id: coupler resonator spectroscopy
      operation: coupler_resonator_spectroscopy
      parameters:
        measured_qubits: 0
        amplitude: 0.1

Parameters
----------

- measured_qubits: qubit to measure from the pair
- amplitude: readout or qubit drive amplitude (optional). If defined, same amplitude will be used in all qubits. Otherwise the default amplitude defined on the platform runcard will be used
