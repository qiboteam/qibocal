.. _example:

Minimal working example
=======================

This section shows the steps to perform a resonator spectroscopy with Qibocal.

Write a runcard
^^^^^^^^^^^^^^^

A runcard contains all the essential information to run a specific task.
For our purposes, we can use the following runcard and save it as ``example.yml`` :

.. code-block:: yaml

    platform: tii5q

    qubits: [2]

    format: csv

    actions:
        resonator_spectroscopy:
            lowres_width: 5_000_000
            lowres_step: 2_000_000
            highres_width: 1_500_000
            highres_step: 200_000
            precision_width: 1_500_000
            precision_step: 100_000
            software_averages: 1

More examples of runcards are available on `Github <https://github.com/qiboteam/qibocal/tree/main/runcards>`_ .

Run the routine
^^^^^^^^^^^^^^^
To run all the calibration routines specified in the ``runcard``, Qibocal uses the ``qq`` command

.. code-block::

    qq <runcard> -o <output_folder>

if ``<output_folder>`` is specified, the results will be saved in it, otherwise ``qq`` will automatically create a default folder containing the current date and the username.

In our case we can use the command:

.. code-block::

    qq example.yml -o resonator_spectroscopy_routine


Visualize the data
^^^^^^^^^^^^^^^^^^

Qibocal gives the possibility to show a live-plotting with the :ref:`qqlive` command. Opening the output link in the browser, the web page will look like the following image.

.. image:: resonator_spectr_qqlive.png
    :align: center



Uploading reports to server
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to upload the report to a centralized server, send to the server administrators your public ssh key (from the machine(s) you are planning to upload the report) and then use the ``qq-upload <output_folder>`` command. This program will upload your report to the server and generate an unique URL.
