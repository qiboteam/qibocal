Calibration routines
--------------------

In ``qibocal`` we provide the following calibration routines:

Resonator Spectroscopy with Attenuation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibocal.calibrations.characterization.resonator_spectroscopy
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
