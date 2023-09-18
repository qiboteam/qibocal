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

Installing with pip
"""""""""""""""""""

The installation using ``pip`` is the recommended approach to install ``Qibocal``.

.. code-block::

      pip install qibocal

.. note::
      Make sure to update ``pip`` if needed.


Installing from source
""""""""""""""""""""""

In order to install ``qibocal`` from source, you can simply clone the GitHub repository
and perform the installation by following these instructions.

.. code-block::

      git clone https://github.com/qiboteam/qibocal.git
      cd qibocal
      pip install .

You can also use `poetry <https://python-poetry.org/>`_ to install ``qibocal`` from source:

.. code-block::

      git clone https://github.com/qiboteam/qibocal.git
      cd qibocal
      poetry install
