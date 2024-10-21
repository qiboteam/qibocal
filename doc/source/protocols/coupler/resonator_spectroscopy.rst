Coupler Resonator Spectroscopy
==============================

This protocol consists on a frequency sweep on the readout frequency while we change the flux coupler pulse amplitude of
the coupler pulse. We expect to enable the coupler during the amplitude sweep and detect an avoided crossing
that will be followed by the frequency sweep. No need to have the qubits at resonance. This should be run after
resonator_spectroscopy to detect couplers and adjust the coupler sweetspot if needed and get some information
on the flux coupler pulse amplitude requiered to enable 2q interactions.

Parameters
^^^^^^^^^^

.. autoclass::
   qibocal.protocols.couplers.coupler_qubit_spectroscopy.CouplerSpectroscopyParameters
   :noindex:
