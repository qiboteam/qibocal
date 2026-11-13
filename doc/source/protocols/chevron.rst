Chevron
=======

In order to implement a two-qubit gate in superconducting quantum computing, it is necessary to bring the two qubits near resonance using specific pulse sequences.
The Chevron protocol implemented in Qibocal can be used to calibrate both CZ and iSWAP gates.

The pulse sequence used to calibrate the iSWAP gate consists of a :math:`\pi` pulse followed by a flux pulse of varying amplitude and duration, applied to the qubit with the highest frequency in the pair.
The initial :math:`\pi` pulse brings the qubit into the state :math:`\ket{1}` while the flux pulse detunes its frequency near resonance with the second qubit. The implementation of the iSWAP gate leverages the avoided crossing between the states :math:`\ket{10}` and :math:`\ket{01}`.

The expected population oscillation pattern follows:

.. math::

	p_e(t, \Delta) = \frac{\Delta^2}{\Delta^2 + 4g^2} + \frac{4g^2}{\Delta^2 + 4g^2}\cos^2\left(\frac{\sqrt{\Delta^2 + 4g^2}}{2}t + \phi \right)


where :math:`\Delta=\omega_1 - \omega_2`, :math:`g` is the coupling constant for the two qubits and :math:`\phi` takes into account flux distortions.

The pulse sequence used to calibrate the CZ gate is the same as the one for the iSWAP gate, with the addition of an initial :math:`\pi` pulse applied to the qubit with the lower frequency so that both qubits are initially prepared in the :math:`\ket{1}`. With this sequence the CZ gate is implemented leveraging the avoided crossing between the state :math:`\ket{11}` and one of the non-computational states :math:`\ket{02}` or :math:`\ket{20}`, where states are labelled as :math:`\ket{n_L n_H}`, with :math:`L` and :math:`H` the qubits with lower and higher frequency, respectively.
A final :math:`\pi` pulse is applied before readout to the qubit occupying level 2 in the non-computational state.

By default, the interaction is assumed to be the :math:`\ket{11}`–:math:`\ket{02}` one, and the final :math:`\pi` pulse is applied to the qubit with the higher frequency.
Setting ``infer_hybridization: true`` instead infers the interaction from the calibration data.
Since only the higher-frequency qubit is fluxed, its frequency follows :math:`f_H(A)` as a function of the flux pulse amplitude :math:`A`, given by the calibrated flux coefficients, and the two resonance conditions are

.. math::

	\ket{11} \leftrightarrow \ket{02}: \quad f_H(A) = f_L - \alpha_H, \qquad
	\ket{11} \leftrightarrow \ket{20}: \quad f_H(A) = f_L + \alpha_L,

with :math:`\alpha` the anharmonicities.
Both are solved analytically, and the resonance reached with the smallest :math:`|A|` determines which qubit receives the final :math:`\pi` pulse.
This requires the ``frequency_01`` and ``frequency_12`` calibrations of both qubits, together with the ``flux_coefficients`` of the higher-frequency qubit, and is ignored for the iSWAP gate.

Parameters
^^^^^^^^^^

.. autoclass::
    qibocal.protocols.two_qubit_interaction.chevron.chevron.ChevronParameters
    :noindex:

Example
^^^^^^^

Below is an example runcard for this experiment.

.. code-block:: yaml

    - id: chevron
      operation: chevron
      parameters:
        amplitude_max: 0.48
        amplitude_min: 0.45
        amplitude_step: 0.0005
        duration_max: 100
        duration_min: 5
        duration_step: 4
        relaxation_time: 50000

The same experiment, with the interaction branch inferred from the calibration data, can be written using the range syntax as

.. code-block:: yaml

    - id: chevron
      operation: chevron
      parameters:
        amplitude: [0.45, 0.48, 0.0005]
        duration: [5, 100, 4]
        infer_hybridization: true
        relaxation_time: 50000

The expected output is the following:

.. image:: chevron.png

The plot represents the probability of measuring qubit 1 in the excited state as a function of the flux pulse parameters.
The characteristic shape of the plot, known as a Chevron pattern, appears as a consequence of the interaction of the two qubits through their coupling, leading to population exchange.

Before running the Chevron routine it may be useful to run a Cryoscope experiment in order to correct possible distortions in the flux pulse.


Chevron with signal
===================

If the readout fidelity is not high enough, it is possible to run the experiment by measuring the acquired signal.
The data are later normalized between 0 and 1. Here is a possible runcard with the corresponding outcome.


.. code-block:: yaml

    - id: chevron_signal
      operation: chevron_signal
      parameters:
        amplitude_max: 0.48
        amplitude_min: 0.45
        amplitude_step: 0.0005
        duration_max: 100
        duration_min: 5
        duration_step: 4
        relaxation_time: 50000

.. image:: chevron_signal.png
