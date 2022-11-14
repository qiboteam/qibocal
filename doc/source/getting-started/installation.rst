Installation instructions
=========================

Operating system support
""""""""""""""""""""""""

In the table below we summarize the status of *pre-compiled binaries
distributed with pypi* for the packages listed above.

+------------------+---------+
| Operating System | qibocal |
+==================+=========+
| Linux x86        |   Yes   |
+------------------+---------+
| MacOS >= 10.15   |   Yes   |
+------------------+---------+
| Windows          |   Yes   |
+------------------+---------+

.. note::
      All packages are supported for Python 3.8 to 3.10

.. _installing-qibocal:

Qibocal
^^^^^^^

Installing from source
""""""""""""""""""""""

In order to install ``qibocal`` from source, you can simply clone the GitHub repository
with:

.. code-block::

      git clone https://github.com/qiboteam/qibocal.git
      cd qibocal
      pip install .

You can also use `poetry <https://python-poetry.org/>`_ to install ``qibocal`` from source:

.. code-block::

      git clone https://github.com/qiboteam/qibocal.git
      cd qibocal
      poetry install


Qibocal needs Qibolab!
""""""""""""""""""""""

The ``qibocal`` package works closely with ``qibolab``.
Indeed, in order to carry out calibrations appropriately, it is essential to be able
to translate appropriate theoretical tests into the corresponding pulse sequences that interpret them in hardware.
In this section we explain how to perform calibration of a device belonging to the `tii platform`.
To do this, it is necessary to have ``qibolab`` properly installed.
To do this procedure correctly, visit the `Qibolab installation page`_, where it is explained how to install the appropriate `extra_dependencies`.



.. _`Qibolab installation page`: https://qibolab.readthedocs.io/en/latest/getting-started/installation.html
