Dispersive-shift qutrit
=======================

Experiment Description
----------------------

Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing the resonator shift produced by the coupling between the resonator and the qubit. Do this both for |1> and |2>.

Example Runcard
---------------

.. code-block::

    - id: dispersive shift
      operation: dispersive_shift_qutrit
      parameters:
        freq_width: 10_000_000
        freq_step: 100_000
        nshots: 10

Parameters
----------

- freq_width: width [Hz] for frequency sweep relative to the readout frequency [Hz]
- freq_step: frequency step for sweep [Hz]
