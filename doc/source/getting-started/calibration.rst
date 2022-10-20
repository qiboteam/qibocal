How to calibrate a TII device?
==============================

The ``qibocal`` package works closely with ``qibolab``. 
Indeed, in order to carry out calibrations appropriately, it is essential to be able 
to translate appropriate theoretical tests into the corresponding pulse sequences that interpret them in hardware.
In this section we explain how to perform calibration of a device belonging to the `tii platform`. 
To do this, it is necessary to have ``qibolab`` properly installed, enabling interaction with the machines. 
In order to install appropriately the lab package please clone its GitHub repository with:

.. code-block::

      git clone https://github.com/qiboteam/qibolab.git
      cd qibocal
      pip install .[tiiq]


Calibration execution
^^^^^^^^^^^^^^^^^^^^^

In the previous section we explained how to write a runcard. Once this step is done, simply run the ``qq`` command as follows:

.. code-block::

    qq <runcard> -o <output_folder>


In this way you will execute the actions specified in ``<runcard>`` and they will be saved in the ``<output_folder>``.