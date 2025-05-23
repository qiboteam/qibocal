Cryoscope experiment
====================

In this section we show how to run a cryoscope experiment using Qibocal.

The goal of the Cryoscope experiment is to reconstruct the shape of the flux pulse sent to the qubit in order to determine correction for signal distortions.
To do this we exploit the dependence of the transition frequency of a transmon qubit on the magnetic flux

.. math:: f_Q(\Phi_Q)\approx \frac{1}{h} \left( \sqrt{8E_J E_C \left| \cos\left(\pi\frac{\Phi_Q}{\Phi_0}\right) \right|} - E_C \right)
    :label: transmon

where :math:`E_C` is the charging energy,  :math:`E_J` is the sum of the Jospehson energies and :math:`\Phi_0` is the flux quantum.
The routine implementation follows the description given in :cite:p:`Cryoscope_20`:

.. _cryoscope:

Cryoscope
---------
The cryoscope experiment consists of a Ramsey-like experiment where a flux pulse is embedded between the two :math:`\pi/2` pulses separated by a fixed time interval :math:`T`.
The first :math:`\pi /2` rotation around the :math:`Y` axis change the qubit state from :math:`\ket{0}` to :math:`\frac{\ket{0}+\ket{1}}{\sqrt{2}}`; then the flux pulse transforms the qubit state to :math:`\frac{\ket{0}+e^{i\phi_\tau}\ket{1}}{\sqrt{2}}` where

.. _phase:

.. math:: \frac{\phi_\tau}{2\pi} = \int_0^T \Delta f_Q(\Phi_{Q,\tau}(t))dt
    :label: phase

Then the experiment is completed with a :math:`\pi/2` rotation either around the :math:`y` axis or around the :math:`x` axis in order to obtain, respectively the :math:`\langle Y \rangle` or  :math:`\langle X \rangle` component of the Bloch vector.
From the :math:`\langle X \rangle` and :math:`\langle Y \rangle` components of the Bloch vector we can derive the relative phase :math:`\phi_\tau` which in turn can be used to computed

.. math::

    \Delta f_R \equiv \frac{\phi{\tau+\Delta\tau} - \phi_{\tau}}{2\pi \Delta\tau}

and then we can extract an estimate of the effective flux pulse :math:`\Phi_Q(t)` on the qubit by inverting :math:numref:`transmon`.


Parameters
^^^^^^^^^^

.. autoclass:: qibocal.protocols.flux_dependence.cryoscope.CryoscopeParameters
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

.. image:: cryoscope.png


.. note::
  In the case where there are no filters the protocol will compute the FIR and the IIR
  filters. If the filters are already present the computation of the filters will be skipped
  and only the reconstructed waveform will be shown.


Requirements
^^^^^^^^^^^^

- :ref:`single-shot`
- :ref:`flux_amplitude`
