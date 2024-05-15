Resonator flux dependence
=========================

Experiment Description
----------------------

Perform a scan in frequency and flux, so that we can see the dependence of the resonator to the external DC flux.
Note that the dependence is indirect, since it is actually only the qubit (SQUID) to be dependet on the external flux.
It is therefore required to use a low-power pulse for the measurement.

Example Runcard
---------------

.. code-block::

    - id: resonator flux dependence
      operation: resonator_flux
      parameters:
        freq_width: 10_000_000
        freq_step: 500_000
        bias_width: 0.8
        bias_step:  0.1
        nshots: 1024
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
