Qubit flux tracking
===================

Experiment Description
----------------------

Perform a scan in frequency and flux, so that we can see the dependence of the resonator to the external DC flux.

Differently from the simple qubit-flux-dependence, the experiment continuously changes the center of the scan, while also varying the flux.

Example Runcard
---------------

.. code-block::

    - id: qubit flux tracking
      operation: qubit_flux_tracking
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
- drive_amplitude: drive amplitude (optional). If defined, same amplitude will be used in all qubits.
- transition: flux spectroscopy transition type ("01" or "02"). Default value is 01
- drive_duration: int = 2000
