.. _stark-shift:

Stark Shift
===========

In general, we implecitly assume that the intra-cavity photon population is zero, if we remove this constraint the qubit frequency will appear shifted.

We can see this looking at the Hamiltonian of the qubit-resonator interaction (:math:`\omega_r` and :math:`\omega_q` are the resonator and qubit frequencies, :math:`g` is the couling between the two, :math:`\Delta` is defined as :math:`\Delta=|\omega_q-\omega_r|`) :cite:`Blais_2004, Krantz_2019`:

.. math::

   H = \omega_r \left( a^\dagger a + \frac{1}{2}\right) + \frac{1}{2}\left(\omega_q + \frac{g^2}{\Delta} + \frac{2g^2}{\Delta}a^\dagger a \right)

Where the last term is usually called "ac-Stark shift".

An increased population in the cavity, wich can be caused by a low relaxation time betwen subsequent shots (if the cavity lifetime is long enough), or by drive powers too high, will therefore lead to qubit frequency (:math:`\omega_q`) shift of :math:`2\chi<a^\dagger a>` by average (with :math:`\chi=\frac{g^2}{\Delta}`).

Eventually, using prior knowledge of the value of :math:`\chi`, it is possile to precisely infer the number of cavity photons, for different drive powers.

In the experiment, implemented in Qibocal, a qubit spectroscopy (see :ref:`qubit-spectroscopy`) will be performed for different drive powers. At low powers, the approximation of zero cavity photons will hold, while at high power it won't, and a shift in frequency will be clearly visible.
The shift, will then increase quadratically with the drive power: :math:`\delta f \approx \frac{\alpha}{2\Delta (\alpha - \Delta)} \Omega^2` (where :math:`\Omega` is the drive amplitude and :math:`\alpha` is the qubit anharmonicity.)

Parameters
^^^^^^^^^^

.. autoclass:: qibocal.protocols.stark_shift.StarkShiftParameters


How to execute a Stark shift experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A possible runcard to launch a Stark shift experiment could be:

.. code-block:: yaml

  - id: stark shift
    operation: starkshift
    parameters:
      freq_width: 5_000_000
      freq_step: 10_000
      min_amp_factor: 0
      max_amp_factor: 1
      step_amp_factor: 0.02
      duration: 6000  # drive duration
      amplitude: 1  # starting drive amplitude
      nshots: 5_000

Here is the corresponding plot:

# TODO: ADD PLOT

Requirements
^^^^^^^^^^^^

- :ref:`resonator_spectroscopy`
- :ref:`qubit-spectroscopy`
