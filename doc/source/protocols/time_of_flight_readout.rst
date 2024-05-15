Time of flight - readout
========================

Experiment Description
----------------------

Computes the time of flight for a readout signal, namely the time required for a synthesized measurement signal to go across the qubit and be acquired by the controller.

Example Runcard
---------------

.. code-block::

    - id: time_of_flight_readout
      operation: time_of_flight_readout
      readout_amplitude: 1
      window_size: 50
      parameters:
        nshots: 1024

Parameters
----------

- readout_amplitude: amplitude of the readout pulse
- window_size: window size for the moving average
