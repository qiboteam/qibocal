Resonator crosstalk
===================

Experiment Description
----------------------

Perform a scan in frequency and flux, so that we can see the dependence of the resonator to the external DC flux.
Other resonator are also monitored during the experiment, so that it is possible to map the crosstalk matrix of the resonators.

Example Runcard
---------------

.. code-block::

    - id: resonator_crosstalk
      operation: resonator_crosstalk
      parameters:
        freq_width: 10_000_000
        freq_step: 1_000_000
        bias_width: 0.8 #0.1
        bias_step:  0.1 # 0.001
        drive_amplitude: 0.005
        nshots: 1000
        relaxation_time: 0

Parameters
----------

- freq_width: width for frequency sweep relative to the readout frequency [Hz]
- freq_step: frequency step for sweep [Hz]
- bias_width: width for bias sweep [V]
- bias_step: bias step for sweep [a.u.]
- flux_amplitude_start: amplitude start value(s) for flux pulses sweep relative to the qubit sweetspot [a.u.]
- flux_amplitude_end: amplitude end value(s) for flux pulses sweep relative to the qubit sweetspot [a.u.]
- flux_amplitude_step: amplitude step(s) for flux pulses sweep [a.u.]
- flux_qubits: IDs of the qubits that we will sweep the flux on. If ``None`` flux will be swept on all qubits that we are running the routine on in a multiplex fashion. If given flux will be swept on the given qubits in a sequential fashion (n qubits will result to n different executions). Multiple qubits may be measured in each execution as specified by the ``qubits`` option in the runcard.
