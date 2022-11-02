Calibration routines
--------------------

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

Routine-specific method for post-processing data acquired.

.. automodule:: qibocal.fitting.methods
   :members:
   :undoc-members:
   :show-inheritance: