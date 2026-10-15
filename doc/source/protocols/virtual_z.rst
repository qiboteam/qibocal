Virtual Z
=========

The protocol applies the CZ gate once and measures conditional oscillations while
sweeping the virtual Z phase.

Parameters
^^^^^^^^^^

.. autoclass::
	qibocal.protocols.two_qubit_interaction.virtual_z_phases.VirtualZPhasesParameters
	:noindex:

Example
^^^^^^^

Below is an example runcard for this experiment.

.. code-block:: yaml

     - id: cz
       operation: correct_virtual_z_phases
       parameters:
        dt: 0
        theta_end: 6.5
        theta_start: 0
        theta_step: 0.5

The expected output is the following:

.. image:: virtual_z.png
