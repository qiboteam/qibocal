Interactive usage
=================

A great use of Qibocal is to run routines interactively, processing the results of the
routines in an interpreter or a notebook, or launching further ones only after receiving
results in the same environment.

An example of this workflow is shown in the `README
<https://github.com/qiboteam/qibocal#minimal-working-example>`_.

How to run it on a queue
------------------------

While Qibocal itself does not require any special handling for interactive usage, it is
common enough to place shared resource behind a queue infrastructure.

In these cases, you may experience some additional complications, since Qibocal has to
run on a machine with the capability of connecting to the control electronics.

To cater for this use case, the only additional requirement is to run your notebook (or
intepreter) on the machine with instrument access, and possibly establish a suitable
port forward for its interface just through ``ssh`` itself.

.. image:: ./ssh-remote-port.png
    :width: 90%
    :align: center

Since it is potentially complex to identify which remote address and port to forward,
the following script should help automating the process.

.. literalinclude:: ./connect-to-queue.bash
   :language: bash

Once the required server is dispatched as queue job with the script, the instructions to
connect will appear in the output (usually redirected by the queue system on a file).
