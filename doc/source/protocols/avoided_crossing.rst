Avoided crossing
================

Experiment Description
----------------------

This routine performs the qubit flux dependency for the "01" and "02" transition on the qubit pair. It returns the bias and frequency values to perform a CZ and a iSwap gate.

Example Runcard
---------------

.. code-block::

    - id: avoided crossing
      operation: avoided_crossing
      parameters:
        drive_amplitude: 0.2
        transition: 01
        drive_duration: 200

Parameters
----------

- drive_amplitude: drive amplitude (optional). If defined, same amplitude will be used in all qubits. Otherwise the default amplitude defined on the platform runcard will be used
- transition: flux spectroscopy transition type ("01" or "02"). Default value is 01
- drive_duration: duration of the drive pulse
