Standard Error Amplification
============================

The goal of the Standard Error Amplification experiment is to estimate the error on the conditional phase acquired by the :math:`CZ` gate, so that this error can be tracked and used to keep the two-qubit gate calibrated over time.
An ideal :math:`CZ` gate applies a phase of :math:`\pi` to the :math:`\ket{11}` component of the two-qubit state and leaves the other computational states unchanged. In practice, the gate implemented on hardware applies a phase :math:`\pi + \delta`, where :math:`\delta` is a small, unknown error that we want to measure.

Since :math:`\delta` is typically too small to be resolved reliably in a single application of the gate, we amplify it by repeating the :math:`CZ` gate many times inside a Ramsey-like interferometer, so that the error accumulates linearly with the number of repetitions and becomes measurable.

.. _sea:

Standard Error Amplification
-----------------------------

Let :math:`Q_a` be the probe qubit and :math:`Q_b` the control qubit of the pair. The experiment consists of the following circuit, repeated :math:`n` times:

.. math:: \big[Y(\pi)_{Q_b}\ X(\pi)_{Q_a}\ CZ\big]^{n-1}\ CZ

sandwiched between an :math:`X(\pi/2)_{Q_a}` pulse at the beginning and an :math:`X(\pi/2)_{Q_a}` pulse at the end, for a total of :math:`2n` applications of the :math:`CZ` gate.

The first :math:`X(\pi/2)` pulse puts :math:`Q_a` on the equator of the Bloch sphere, :math:`\frac{\ket{0}-i\ket{1}}{\sqrt{2}}`, the standard first pulse of a Ramsey sequence: from now on, :math:`Q_a` stores any relative phase it accumulates as a rotation around the equator.
Since :math:`Q_b` starts in :math:`\ket{0}`, the first :math:`CZ` is inactive: with :math:`Q_b=\ket{0}` the gate acts trivially and applies no conditional phase. The subsequent :math:`Y(\pi)` pulse on :math:`Q_b` is a hard bit flip that brings :math:`Q_b` to :math:`\ket{1}`, so that the next :math:`CZ` becomes active and genuinely imprints the conditional phase :math:`\pi+\delta` onto the part of :math:`Q_a`'s superposition correlated with :math:`Q_b=1`. The :math:`X(\pi)` pulse applied to :math:`Q_a` at the same time is an echo: it does not carry any information by itself, but it swaps the physical levels of :math:`Q_a`, so that any unwanted single-qubit phase acquired at every :math:`CZ` — active or not — cancels out pairwise over the sequence, in the same way a spin echo removes a static detuning. The :math:`Y(\pi)` pulses on :math:`Q_b` play the equivalent role for :math:`Q_b`'s own single-qubit phase, while simultaneously toggling the :math:`CZ` between active and inactive.

As the sequence is stepped through, the :math:`2n` applications of :math:`CZ` therefore alternate inactive, active, inactive, active, :math:`\dots`, always ending on an active one. Out of the :math:`2n` total gates, exactly :math:`n` are active, and each active gate contributes :math:`\pi + \delta` to :math:`Q_a`'s accumulated relative phase, while the single-qubit phase contributions cancel thanks to the echo. The net phase accumulated by :math:`Q_a` after the full sequence is, up to known and calculable offsets,

.. math:: \phi_n = n\,\delta
    :label: sea_phase

The final :math:`X(\pi/2)` pulse on :math:`Q_a` is the second pulse of the Ramsey pair: it converts the phase :math:`\phi_n` of :eq:`sea_phase` into a measurable excited-state population

.. math:: P(\ket{1}_{Q_a}) \approx \sin^2\left(\frac{n\,\delta}{2}\right)
    :label: sea_population

By measuring :math:`P(\ket{1}_{Q_a})` as a function of the number of repetitions :math:`n` and fitting it to a sinusoid, we extract :math:`\delta`, the error on the :math:`CZ` conditional phase.

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

.. image::

.. note::
      The mitigated line will appear if the readout mitigation matrix is available in the platform calibration.
      This can be obtained using the :ref:`readout-mitigation-matrix` routine.


Requirements
^^^^^^^^^^^^

- :ref:`cz_amplitude`
