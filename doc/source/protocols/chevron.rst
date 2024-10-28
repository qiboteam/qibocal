Chevron
=======

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
        amplitude_max_factor: 1.1
        amplitude_min_factor: 0.9
        amplitude_step_factor: 0.01
        duration_max: 51
        duration_min: 4
        duration_step: 2

The expected output is the following:

.. image:: chevron.png
