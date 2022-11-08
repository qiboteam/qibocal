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

.. automodule:: qibocal.calibrations.protocols.randomized_benchmarking
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

In ``qibocal`` all the data are stored using the :class:`qibocal.data.Dataset` which provide
different formats for storing the data including `pickle <https://docs.python.org/3/library/pickle.html>`_
and `csv <https://docs.python.org/3/library/csv.html>`_.

.. autoclass:: qibocal.data.Dataset
    :members:
    :member-order: bysource
