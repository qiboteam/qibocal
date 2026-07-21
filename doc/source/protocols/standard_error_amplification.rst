Standard Error Amplification
============================

The goal of the Standard Error Amplification experiment is to estimate the error on the conditional phase acquired by the :math:`CZ` gate.

An ideal :math:`CZ` gate applies a phase of :math:`\pi` to the :math:`\ket{11}` component of the two-qubit state and leaves the other computational states unchanged.
In practice, the gate implemented on hardware applies a phase :math:`\pi + \delta`, where :math:`\delta` is a small, unknown error.

The implementation of the protocol follws the on described in :cite:p:`sea`.

Standard Error Amplification
-----------------------------

Let :math:`Q_a` be the probe qubit and :math:`Q_b` the control qubit of the pair.

The experiment consists of the following circuit:

..
  missing svg image of the circuit

The net phase accumulated by :math:`Q_a` after the full sequence is, up to known and calculable offsets,

.. math:: \phi_n = n\,\delta .
    :label: sea_phase

The final :math:`X(\pi/2)` pulse on :math:`Q_a` is the second pulse of the Ramsey pair: it converts the phase :math:`\phi_n` of :eq:`sea_phase` into a measurable excited-state population

.. math:: P(\ket{1}_{Q_a}) \approx \sin^2\left(\frac{n\,\delta}{2}\right)
    :label: sea_population

By measuring :math:`P(\ket{1}_{Q_a})` as a function of the number of repetitions :math:`n` and fitting it to a sinusoid, it is possible to extract :math:`\delta`.

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


Requirements
^^^^^^^^^^^^

- :ref:`cz_amplitude`
