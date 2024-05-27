Time Of Flight (Readout)
========================

In this section, we present the time-of-flight experiment for qibocal.

In the `time of flight` experiment, we measure the time it takes for a readout pulse to travel to the qubit and back, being acquired by the control instrument ADCs.

Carefully calibrating this delay time is important to optimize the readout length. In particular, it is useful to acquire for a duration equal to that of the pulse, where it is possible to see differences (in amplitude and phase) between the two states of a qubit.

The parameters for the experiment are :class:`qibocal.protocols.signal_experiments.time_of_flight_readout.TimeOfFlightReadoutParameters`, namely the amplitude of the readout pulse (if it is not set, the runcard one is used) and an integer that is used as the window size for a moving average.

How to Execute an Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    - id: time of flight experiment      # custom ID of the experiment
      operation: time_of_flight_readout  # unique name of the routine
      parameters:
        readout_amplitude: 1             # usually high
        window_size: 10
        nshots: 1024
        relaxation_time: 20_000

Although it is possible to avoid setting a specific readout amplitude, it is generally useful to set a high value here. Indeed, we are not looking for the optimal amplitude but want to have a signal with enough power so that it is clear when it starts.

Acquisition
^^^^^^^^^^^

The acquisition procedure is described in :func:`qibocal.protocols.signal_experiments.time_of_flight_readout._acquisition`. It is important to note that this experiment makes use of the RAW acquisition mode, which may require some specific care depending on the instrument employed.

TODO: ADD PLOT

Fit
^^^

The fit procedure (:func:`qibocal.protocols.signal_experiments.time_of_flight_readout._fit`) employs a moving average, returning the time when it is maximum, namely when the signal starts being acquired.
