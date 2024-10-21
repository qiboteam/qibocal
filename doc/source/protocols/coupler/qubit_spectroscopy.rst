Coupler Qubit Spectroscopy
==========================

This protocol consists on a frequency sweep on the qubit frequency while we change the flux
coupler pulse amplitude of the coupler pulse.
We expect to enable the coupler during the amplitude sweep and detect an avoided
crossing that will be followed by the frequency sweep.
This needs the qubits at resonance, the routine assumes a sweetspot value for
the higher frequency qubit that moves it to the lower frequency qubit instead
of trying to calibrate both pulses at once. This should be run after qubit_spectroscopy
to further adjust the coupler sweetspot if needed and get some information on
the flux coupler pulse amplitude requiered to enable 2q interactions.

Parameters
^^^^^^^^^^

.. autoclass::
   qibocal.protocols.couplers.coupler_qubit_spectroscopy.CouplerSpectroscopyParametersQubit
   :noindex:
