.. _calibration_routines:


Protocols
=========

The calibration routines are techniques used to reduce execution errors in quantum circuits.
To perform calibrations, the expected theoretical results of an experiment are compared with readouts obtained by
running that same experiment using quantum devices. There are at least two objectives in calibration:

- To derive the exact values of the signals to be supplied to the hardware in order to obtain the best
  possible implementation of a theoretical operation;
- Derive the system relaxation time, which is the threshold time interval beyond which executions of operations
  on the hardware are no longer reliable due to machine overload.


Resonator spectroscopy
----------------------

Resonator spectroscopy sweeping amplitude
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.resonator_spectroscopy
   :members:
   :undoc-members:
   :show-inheritance:

Resonator spectroscopy sweeping attenuation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.resonator_spectroscopy_attenuation
   :members:
   :undoc-members:
   :show-inheritance:


Resonator punchout
------------------


Punchout sweeping amplitude
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.resonator_punchout
   :members:
   :undoc-members:
   :show-inheritance:

Punchout sweeping attenuation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.resonator_punchout_attenuation
   :members:
   :undoc-members:
   :show-inheritance:



Qubit spectroscopy
------------------

Qubit spectroscopy protocol
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.qubit_spectroscopy
   :members:
   :undoc-members:
   :show-inheritance:


Flux dependence experiments
---------------------------

Resonator dependence with flux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. automodule:: qibocal.protocols.characterization.flux_dependence.resonator_flux_dependence
   :members:
   :undoc-members:
   :show-inheritance:


Qubit dependence with flux
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.flux_dependence.qubit_flux_dependence
   :members:
   :undoc-members:
   :show-inheritance:


Rabi experiments
----------------

Rabi amplitude
^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.rabi.amplitude
   :members:
   :undoc-members:
   :show-inheritance:

Rabi length
^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.rabi.length
   :members:
   :undoc-members:
   :show-inheritance:

Ramsey experiments
------------------

Ramsey with sweeper
^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.ramsey
   :members:
   :undoc-members:
   :show-inheritance:

Ramsey without sweeper
^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: qibocal.protocols.characterization.ramsey_sequences
   :members:
   :undoc-members:
   :show-inheritance:


Coherence experiments
---------------------

T1
^^
.. automodule:: qibocal.protocols.characterization.coherence.t1
   :members:
   :undoc-members:
   :show-inheritance:

T1 (using pulse sequences)
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: qibocal.protocols.characterization.coherence.t1_sequences
   :members:
   :undoc-members:
   :show-inheritance:

T2
^^
.. automodule:: qibocal.protocols.characterization.coherence.t2
   :members:
   :undoc-members:
   :show-inheritance:

T2 (using pulse sequences)
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: qibocal.protocols.characterization.coherence.t2_sequences
   :members:
   :undoc-members:
   :show-inheritance:


Spin Echo
^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.coherence.zeno
   :members:
   :undoc-members:
   :show-inheritance:


Single shot classification
--------------------------

Classification experiment
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.classification
   :members:
   :undoc-members:
   :show-inheritance:


AllXY and Drag pulse tuning
---------------------------


AllXY
^^^^^

.. automodule:: qibocal.protocols.characterization.allxy.allxy
   :members:
   :undoc-members:
   :show-inheritance:


AllXY (multiple beta values)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.allxy.allxy_drag_pulse_tuning
   :members:
   :undoc-members:
   :show-inheritance:

Drag pulse tuning
^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.allxy.drag_pulse_tuning
   :members:
   :undoc-members:
   :show-inheritance:


Readout optimization
--------------------

Fine-tuning of the readout pulse frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.readout_optimization.resonator_frequency
   :members:
   :undoc-members:
   :show-inheritance:

Fast-Reset
^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.fast_reset.fast_reset
   :members:
   :undoc-members:
   :show-inheritance:


Signal experiments
------------------

Time of flight
^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.signal_experiments.time_of_flight_readout
   :members:
   :undoc-members:
   :show-inheritance:

Readout characterization experiments
------------------------------------


Computing fidelity and QND
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.readout_characterization
   :members:
   :undoc-members:
   :show-inheritance:


Two qubit gates experiments
---------------------------


Chevron
^^^^^^^

.. automodule:: qibocal.protocols.characterization.two_qubit_interaction.chevron
   :members:
   :undoc-members:
   :show-inheritance:

Tune Landscape
^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.two_qubit_interaction.cz_virtualz
   :members:
   :undoc-members:
   :show-inheritance:


Fitting functions
-----------------

Classifiers
^^^^^^^^^^^
.. automodule:: qibocal.fitting.classifier.run
   :members:
   :undoc-members:
   :show-inheritance:

Statistics
^^^^^^^^^^
.. automodule:: qibocal.bootstrap
   :members:
   :undoc-members:
   :show-inheritance:


Gate set characterization
-------------------------

.. _abstract-module-label:

Abstract and Basic methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

Fitting methods
"""""""""""""""
.. automodule:: qibocal.protocols.characterization.randomized_benchmarking.fitting
   :members:
   :undoc-members:
   :show-inheritance:

Prebuild noise models
"""""""""""""""""""""
.. automodule:: qibocal.protocols.characterization.randomized_benchmarking.noisemodels
   :members:
   :undoc-members:
   :show-inheritance:
