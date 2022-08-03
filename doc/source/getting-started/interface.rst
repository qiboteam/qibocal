Interface
=========

In this section we present the different commands implemented in ``qcvv`` and how to use them.

``qq``
^^^^^^
``qq`` is the base command in ``qcvv``. It can be launched from the command line using:

.. code-block::

    qq <runcard> -o <output_folder>

It will run all the calibration routines specified in the ``<runcard>`` file and save all results
in the ``<output_folder>``. The runcard layout is specified in the :ref:`this <runcard>` section.
If no ``<output_folder>`` is specified ``qq`` will automatically create a default folder containing
the current date and the username.


``qq-live``
^^^^^^^^^^^

``qq-live`` is the command dedicated to live-plotting. It can be launched from the command line using:

.. code-block::

    qq-live <output_folder>

where ``<output_folder>``  is the directory where all the measurements are saved after running ``qq``.
``qq-live`` will produce specific plots depending on the calibration routines executed. If the ``qq`` command
is still running ``qq-live`` will instead produce a live-plotting.

``qq-upload``
^^^^^^^^^^^^^

``qq-compare``
^^^^^^^^^^^^^^
