Coupler chevron
===============

Experiment Description
----------------------

Perform an CZ experiment between pairs of qubits by changing the coupler state, qubits need to be pulses into their interaction point.

Example Runcard
---------------

.. code-block::

    - id: coupler chevron
      operation: coupler_chevron
      parameters:
        measured_qubits: 0
        amplitude: 0.1
        native_gate: CZ

Parameters
----------

- measured_qubits: qubit to measure from the pair
- amplitude: readout or qubit drive amplitude (optional). If defined, same amplitude will be used in all qubits. Otherwise the default amplitude defined on the platform runcard will be used
- native_gate: native gate to implement, CZ or iSWAP
