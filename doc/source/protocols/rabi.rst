Rabi experiment
==============

The goal of the Rabi experiment is to tune the amplitude (or the time) of the drive pulse, in order
to move the qubit state from the ground to the first excited one.

In the Rabi experiment, the qubit is probed with a drive pulse at the qubit frequency :math:`w_{01}`
before measuring. This pulse sequence is repeated multiple times changing the amplitude (the time).
The qubit starts in the ground state, changing one of the two parameters of the drive pulse, the probability of being in the excited state increases following a sinusoidal pattern.

For the amplitude version, we expect:

.. math::
	p_e(t) = sin(\Omega_R \frac{t}{2})

For the time version, we have to take into account the dephasing and the energy decay. In case the
Rabi rate is larger than the decay and the pure dephasing rate,

.. math::
	p_e(t) = \frac{1}{2} (1- e^{-t/\tau} sin(\Omega_R \frac{t}{2}))

where :math:`\Omega_R` is the Rabi frequency and :math:`\tau` the decay time.

Example
^^^^^^^
It follows an example of the experiment parameters.

.. code-block:: yaml

    - id: Rabi
      operation: rabi_amplitude
      parameters:
	  	min_amp_factor: 0.1
		max_amp_factor: 1.
		step_amp_factor: 0.01
		pulse_length: 40
		nshots: 3000

A detailed explanation of the parameters can be found in :class:`qibocal.protocols.rabi.amplitude.RabiAmplitudeParameters` or `qibocal.protocols.rabi.length.RabiAmplitudeParameters`.
