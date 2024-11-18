Flipping
========

The flipping experiment corrects the amplitude in the qubit drive pulse. In this experiment,
we applying an :math:`R_x(\pi/2)` rotation followed by :math:`N` flips (two :math:`R_x(\pi)` rotations)
and we measure the qubit state.
The first :math:`R_x(\pi/2)` is necessary to discriminate the over rotations and under rotations of the :math:`R_x(\pi)` pulse:
without it the difference between the two cases is just a global phase, i.e., the
probabilities are the same. With the :math:`R_x(\pi/2)` pulse, in case of under rotations the state will be closer to :math:`\ket{0}`
after the initial flip, in the over rotations one the final state will be closer to :math:`\ket{1}`.

By fitting the resulting data with a sinusoidal function, we can determine the delta amplitude, which allows us to refine the
:math:`\pi` pulse amplitue.

Parameters
^^^^^^^^^^

.. autoclass:: qibocal.protocols.flipping.FlippingParameters
	:noindex:

Example
^^^^^^^
It follows a runcard example of this experiment.

.. code-block:: yaml

	- id: flipping
	  operation: flipping
	  parameters:
	    delta_amplitude: 0.05
	    nflips_max: 30
	    nflips_step: 1

The expected output is the following:

.. image:: flipping.png

Requirements
^^^^^^^^^^^^

- :ref:`rabi`
- :ref:`single-shot`
