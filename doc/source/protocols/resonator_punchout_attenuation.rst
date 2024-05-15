Resonator_punchout attenuation
==============================

Experiment Description
----------------------

Experiment to find the resonator frequency: scanning the frequencies while also changing the amplitude (attenuation).

Example Runcard
---------------

.. code-block::

    - id: resonator_punchout_attenuation
      operation: resonator_punchout_attenuation
      parameters:
        freq_width: 10_000_000
        freq_step: 500_000
        min_att: 4
        max_att: 60
        step_att: 4
        nshots: 1000
        relaxation_time: 0

Parameters
----------

- freq_width: width for frequency sweep relative  to the readout frequency [Hz]
- freq_step: frequency step for sweep [Hz].
- min_att: attenuation minimum value [dB].
- max_att: attenuation maximum value [dB].
- step_att: attenuation step [dB].
