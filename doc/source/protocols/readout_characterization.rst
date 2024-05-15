Readout characterization experiment
===================================

Experiment Description
----------------------

Measure some interesting parameters about the quality of measurements, QND-ness and effective qubit temperature.

Example Runcard
---------------

.. code-block::

    - id: readout_characterization
      operation: readout_characterization
      parameters:
        nshots: 1024
        delay: 10_000

Parameters
----------

- delay: delay between readouts
