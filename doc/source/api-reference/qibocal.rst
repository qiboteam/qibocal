.. _calibration_routines:

Calibration routines
--------------------

Introduction
^^^^^^^^^^^^

The calibration routines are techniques used to reduce execution errors in quantum circuits. To perform calibrations, the expected theoretical results of an experiment are compared with readouts obtained by running that same experiment using quantum devices. There are at least two objectives in calibration:

- To derive the exact values of the signals to be supplied to the hardware in order to obtain the best possible implementation of a theoretical operation;
- Derive the system relaxation time, which is the threshold time interval beyond which executions of operations on the hardware are no longer reliable due to machine overload.

Current and future ``qibocal`` routines are listed in the following table:

+------------------------+-------------+------------------+---------------+
|       Methods          |     Code    | Tested on device |   Automated   |
+========================+=============+==================+===============+
| Resonator Punchout     |     Yes     |       Yes        |  In progress  |
+------------------------+-------------+------------------+---------------+
| Resonator-Flux         |     Yes     |       Yes        |  In progress  |
+------------------------+-------------+------------------+---------------+
| Qubit spectroscopy     |     Yes     |       Yes        |  In progress  |
+------------------------+-------------+------------------+---------------+
| Qubit Flux             |     Yes     |       Yes        |  In progress  |
+------------------------+-------------+------------------+---------------+
| Rabi (T1)              |     Yes     |       Yes        |       No      |
+------------------------+-------------+------------------+---------------+
| Ramsey (T2)            |     Yes     |       Yes        |       No      |
+------------------------+-------------+------------------+---------------+
| Single shot readouts   |     Yes     |       Yes        |       No      |
+------------------------+-------------+------------------+---------------+
| All-XY (drag)          |     Yes     |       Yes        |       No      |
+------------------------+-------------+------------------+---------------+
| RB & co (1 qubit)      |     Yes     |        No        |       No      |
+------------------------+-------------+------------------+---------------+
| Cryoscope              | In progress |    In progress   |       No      |
+------------------------+-------------+------------------+---------------+
| CZ gate                |      No     |        No        |       No      |
+------------------------+-------------+------------------+---------------+
| RB & co (multi qubits) |      No     |        No        |       No      |
+------------------------+-------------+------------------+---------------+
| Circuit transpilation  |     Yes     |        No        |       No      |
+------------------------+-------------+------------------+---------------+


Resonator
^^^^^^^^^

.. automodule:: qibocal.calibrations.characterization.resonator_spectroscopy
   :members:
   :undoc-members:
   :show-inheritance:

Single Qubit
^^^^^^^^^^^^

Qubit spectroscopy
""""""""""""""""""
.. automodule:: qibocal.calibrations.characterization.qubit_spectroscopy
   :members:
   :undoc-members:
   :show-inheritance:

Rabi oscillations
"""""""""""""""""
.. automodule:: qibocal.calibrations.characterization.rabi
   :members:
   :undoc-members:
   :show-inheritance:

Ramsey
""""""
.. automodule:: qibocal.calibrations.characterization.ramsey
   :members:
   :undoc-members:
   :show-inheritance:

T1
""
.. automodule:: qibocal.calibrations.characterization.t1
   :members:
   :undoc-members:
   :show-inheritance:

Flipping
""""""""
.. automodule:: qibocal.calibrations.characterization.flipping
   :members:
   :undoc-members:
   :show-inheritance:

All-XY
""""""
.. automodule:: qibocal.calibrations.characterization.allXY
   :members:
   :undoc-members:
   :show-inheritance:

State calibration
"""""""""""""""""
.. automodule:: qibocal.calibrations.characterization.calibrate_qubit_states
   :members:
   :undoc-members:
   :show-inheritance:


Data structure
--------------

In ``qibocal`` there are two different objects to manipulate data: :class:`qibocal.data.DataUnits` and :class:`qibocal.data.Data`.

:class:`qibocal.data.DataUnits` is used to store physical related quantities, such as voltages and frequencies. It is a wrapper to a
`pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ where the units of measure for each quantity
are implemented using `pint <https://pint.readthedocs.io/en/stable/>`_.

:class:`qibocal.data.Data` can be used manipulate non-physical quantities.


They provide different formats for storing data including `pickle <https://docs.python.org/3/library/pickle.html>`_
and `csv <https://docs.python.org/3/library/csv.html>`_.

.. autoclass:: qibocal.data.DataUnits
    :members:
    :member-order: bysource

.. autoclass:: qibocal.data.Data
    :members:
    :member-order: bysource


Fitting functions
-----------------

``Qibocal`` offers routine-specific method for post-processing analysis of the data generated by the different :ref:`calibration_routines`.


.. automodule:: qibocal.fitting.methods
   :members:
   :undoc-members:
   :show-inheritance:

Classifiers
^^^^^^^^^^^
.. automodule:: qibocal.fitting.classifier.run
   :members:
   :undoc-members:
   :show-inheritance:

Gate set characterization
-------------------------

.. _abstract-module-label:

Abstract and Basic methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

Circuit Factory
"""""""""""""""
.. automodule:: qibocal.calibrations.niGSC.basics.circuitfactory
   :members:
   :undoc-members:
   :show-inheritance:

Experiment
""""""""""
.. automodule:: qibocal.calibrations.niGSC.basics.experiment
   :members:
   :undoc-members:
   :show-inheritance:

Fitting methods
"""""""""""""""
.. automodule:: qibocal.protocols.characterization.randomized_benchmarking.fitting
   :members:
   :undoc-members:
   :show-inheritance:

Plotting methods
""""""""""""""""
.. automodule:: qibocal.calibrations.niGSC.basics.plot
   :members:
   :undoc-members:
   :show-inheritance:

Useful functions
""""""""""""""""
.. automodule:: qibocal.calibrations.niGSC.basics.utils
   :members:
   :undoc-members:
   :show-inheritance:

Prebuild noise models
"""""""""""""""""""""
.. automodule:: from qibocal.protocols.characterization.randomized_benchmarking.noisemodels
   :members:
   :undoc-members:
   :show-inheritance:


Standard RB
^^^^^^^^^^^
.. automodule:: qibocal.calibrations.niGSC.standardrb
   :members:
   :undoc-members:
   :show-inheritance:

Simultaneous Filtered RB
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: qibocal.calibrations.niGSC.simulfilteredrb
   :members:
   :undoc-members:
   :show-inheritance:

X-ID RB
^^^^^^^
.. automodule:: qibocal.calibrations.niGSC.XIdrb
   :members:
   :undoc-members:
   :show-inheritance:
