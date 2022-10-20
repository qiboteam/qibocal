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

Installing with pip
"""""""""""""""""""

Installing with conda
"""""""""""""""""""""

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
