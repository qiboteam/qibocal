.. _calibration_routinestwo_qubit

Calibration routines
====================

Introduction
^^^^^^^^^^^^

The calibration routines are techniques used to reduce execution errors in quantum circuits.
To perform calibrations, the expected theoretical results of an experiment are compared with readouts obtained by
running that same experiment using quantum devices. There are at least two objectives in calibration:

- To derive the exact values of the signals to be supplied to the hardware in order to obtain the best
  possible implementation of a theoretical operation;
- Derive the system relaxation time, which is the threshold time interval beyond which executions of operations
  on the hardware are no longer reliable due to machine overload.



Single Qubit Characterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Resonator spectroscopy
^^^^^^^^^^^^^^^^^^^^^^

Standard resonator spectroscopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qibocal.protocols.characterization.resonator_spectroscopy
   :members:
   :undoc-members:
   :show-inheritance:

Resonator spectroscopy over attenuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qibocal.protocols.characterization.resonator_spectroscopy_attenuation
   :members:
   :undoc-members:
   :show-inheritance:


Resonator punchout
^^^^^^^^^^^^^^^^^^


Standard resonator spectroscopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qibocal.protocols.characterization.resonator_punchout
   :members:
   :undoc-members:
   :show-inheritance:

Resonator spectroscopy over attenuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qibocal.protocols.characterization.resonator_punchout_attenuation
   :members:
   :undoc-members:
   :show-inheritance:




Qubit spectroscopy
^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.protocols.characterization.qubit_spectroscopy
   :members:
   :undoc-members:
   :show-inheritance:


Rabi experiments
^^^^^^^^^^^^^^^^

Rabi amplitude
~~~~~~~~~~~~~~

.. automodule:: qibocal.protocols.characterization.rabi.amplitude
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qibocal.protocols.characterization.rabi.length
   :members:
   :undoc-members:
   :show-inheritance:



Two Qubit
^^^^^^^^^


Chevron SWAP
""""""""""""
.. automodule:: qibocal.protocols.characterization.two_qubit_interaction.chevron
   :members:
   :undoc-members:
   :show-inheritance:

Tune Landscape
""""""""""""""
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
