.. _Calibration routines:

Calibration routines
--------------------

The calibration routines are techniques used to reduce execution errors in quantum circuits. To perform calibrations, the expected theoretical results of an experiment are compared with readouts obtained by running that same experiment using quantum devices. There are at least two objectives in calibration:

- To derive the exact values of the signals to be supplied to the hardware in order to obtain the best possible implementation of a theoretical operation;
- Derive the system relaxation time, which is the threshold time interval beyond which executions of operations on the hardware are no longer reliable due to machine overload.

In ``qibocal`` we provide the following calibration routines:


.. automodule:: qibocal.calibrations.characterization.resonator_spectroscopy
   :members:
   :undoc-members:
   :show-inheritance:


.. automodule:: qibocal.calibrations.characterization.qubit_spectroscopy
   :members:
   :undoc-members:
   :show-inheritance:


.. automodule:: qibocal.calibrations.characterization.rabi_oscillations
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qibocal.calibrations.characterization.ramsey
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qibocal.calibrations.characterization.t1
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qibocal.calibrations.characterization.flipping
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qibocal.calibrations.characterization.allXY
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qibocal.calibrations.characterization.calibrate_qubit_states
   :members:
   :undoc-members:
   :show-inheritance:


Utils
^^^^^

.. automodule:: qibocal.calibrations.characterization.utils
   :members:
   :undoc-members:
   :show-inheritance:


Data structure
--------------

In ``qibocal`` all the data are stored using the :class:`qibocal.data.DataUnits` and :class:`qibocal.data.Data` which provide
different formats for storing the data including `pickle <https://docs.python.org/3/library/pickle.html>`_
and `csv <https://docs.python.org/3/library/csv.html>`_.

.. autoclass:: qibocal.data.DataUnits
    :members:
    :member-order: bysource

.. autoclass:: qibocal.data.Data
    :members:
    :member-order: bysource
