Cryoscope experiment
==================

In this section we show how to run a cryoscope experiment using Qibocal

.. _cryoscope:

Cryoscope
------
Cryoscope consists of a Ramsey-like experiment where a flux pulse is embedded between the two :math:`\pi/2` pulses.
The first :math:`\pi /2` pulse change the qubit state from :math:`\ket{0}` to :math:`\frac{\ket{0}+\ket{1}}{\sqrt{2}}`; then the flux pulse transforms the qubit state to :math:`\frac{\ket{0}+e^{i\phi_\tau}\ket{1}}{\sqrt{2}}` where

.. math::

  \phi_tau

Then the experiment is completed with a :math:`\pi/2` rotation either around the :math:`y` axis or around the :math:`x` axis in order to obtain, respectively the :math:`\langle Y \rangle` or  :math:`\langle X \rangle` component of the Bloch vector.
From the :math:`\langle X \rangle` and :math:`\langle Y \rangle` components of the Bloch vector we can derive the relative phase :math:`\phi_\tau` and then extract an estimate of the effective flux pulse :math:`\Phi_Q(t)` on the qubit.
The code implementation of the routine follows the description given in :cite:p:`Cryoscope_20`:




Flux amplitude frequency
------




Parameters
^^^^^^^^^^
.. autoclass:: qibocal.protocols.flux_amplitude_frequency.FluxAmplitudeFrequencyParameters
  :noindex:


Example
^^^^^^^
The following is an example of runcard that can be used to acquire the coefficients for the amplitude-frequency relation for the flux pulse

.. code-block:: yaml

  - id: flux amplitude frequency

    operation: flux_amplitude_frequency
    parameters:
      amplitude_max: 0.8
      amplitude_min: 0.0
      amplitude_step: 0.001
      duration: 60
      relaxation_time: 50000


The expected output is the following:
.. image::

Parameters
^^^^^^^^^^

.. autoclass:: qibocal.protocols.two_qubit_interaction.cryoscope.CryoscopeParameters
  :noindex:


Example
^^^^^^^

A possible runcard to launch a Cryoscope experiment could be the following:

.. code-block:: yaml

  - id: cryoscope

    operation: cryoscope
    parameters:
      duration_max: 80
      duration_min: 1
      duration_step: 1
      flux_pulse_amplitude: 0.7
      relaxation_time: 50000


The expected output is the following:

.. image::


Requirements
^^^^^^^^^^^^

- :ref:`single-shot`
