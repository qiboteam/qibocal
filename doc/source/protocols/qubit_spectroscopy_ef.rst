Qubit spectroscopy EF
=====================

Experiment Description
----------------------

Similar to a qubit spectroscopy with the difference that the qubit is first excited to the state 1. This protocols aims at finding the transition frequency between state 1 and the state 2. The anharmonicity is also computed.

If the RX12 frequency is not present in the runcard the sweep is performed around the qubit drive frequency shifted by DEFAULT_ANHARMONICITY, an hardcoded parameter.

Example Runcard
---------------

.. code-block::

    - id: qubit spectroscopy_ef
      operation: qubit_spectroscopy_ef
      parameters:
        drive_amplitude: 0.005
        drive_duration: 2000
        freq_width: 10_000_000
        freq_step: 100_000
        nshots: 1024
        relaxation_time: 0

Parameters
----------

- freq_width: width [Hz] for frequency sweep relative  to the qubit frequency
- freq_step: frequency [Hz] step for sweep
- drive_duration: drive pulse duration [ns]. Same for all qubits
- drive_amplitude: optional amplitude for all pulses
