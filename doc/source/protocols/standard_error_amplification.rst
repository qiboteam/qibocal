.. _standard_error_amplification:

Standard Error Amplification
============================

The goal of the Standard Error Amplification experiment is to estimate the error on the conditional phase acquired during the :math:`CZ` gate.

An ideal :math:`CZ` gate applies a phase of :math:`\pi` to the :math:`\ket{11}` component of the two-qubit state and leaves the other computational states unchanged.

In practice, the gate implemented on hardware applies a phase :math:`\pi + \delta`, where :math:`\delta` is a small, unknown error.

The implementation of the protocol follows the one described in :cite:p:`sea`.

Let :math:`Q_a` be the probe qubit and :math:`Q_b` the other qubit of the pair.

The circuit applies an :math:`X(\pi/2)` pulse on :math:`Q_a`, followed by :math:`2n` :math:`CZ` gates interleaved with :math:`X` on :math:`Q_a` and :math:`Y` on :math:`Q_b`,
and a final :math:`R(\pi/2, \phi_f)` pulse on :math:`Q_a` before measuring it, with :math:`\phi_f = n\pi`.


Since :math:`Q_b` alternates between :math:`\ket{0}` and :math:`\ket{1}`, only :math:`n` of the :math:`2n` :math:`CZ` gates are active,
while the :math:`X` echoes on :math:`Q_a` cancel the single-qubit phases and make the active contributions add up.
:math:`Q_a` therefore accumulates the phase :math:`n(\pi + \delta)` and, for :math:`n \geq 1`,

.. math::
    :label: sea_population_ideal

    P(\ket{1}_{Q_a}) = \sin^2\left(\frac{n(\pi + \delta) - \phi_f}{2}\right) = \sin^2\left(\frac{n\,\delta}{2}\right),

where the choice :math:`\phi_f = n\pi` removes the :math:`(-1)^n` alternation due to the ideal phase :math:`n\pi`.

Decoherence drives :math:`Q_a` towards a mixed state, so the measured probabilities are fitted with

.. math::
    :label: sea_population

    P(\ket{1}_{Q_a}) = \frac{1}{2} + B - \frac{A}{2}\, e^{-\gamma n} \cos(n\,\delta),

where :math:`A` is the contrast, :math:`B` an offset accounting for readout asymmetries and :math:`\gamma` the decay rate of the contrast per repetition.

Parameters
^^^^^^^^^^

.. autoclass::
	qibocal.protocols.two_qubit_interaction.sea.StandardErrorAmplificationParameters
	:noindex:

Example
^^^^^^^
It follows a runcard example of this experiment.

.. code-block:: yaml

    - id: standard_error_amplification
      operation: standard_error_amplification
      targets: [[0, 1]]
      parameters:
        nshots: 2000
        repetitions_max: 10
        repetitions_step: 1

The expected output is the following:


Requirements
^^^^^^^^^^^^

- :ref:`cz_amplitude`
