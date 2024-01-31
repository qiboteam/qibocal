Adding validation to protocols
==============================

Validation of a single node
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The validation step can be added to a node in the runcard by adding an entry ``validator``.

Here is an example of a runcard which validates the results of a T1 experiment
by checking if the value of the :math:`\chi^2` obtained from the fit is below the
threshold 0.5.

.. code-block:: yaml

   actions:
    - id: t1
      priority: 0
      operation: t1
      validator:
        scheme: chi2
        parameters:
            thresholds: [0.5]
      parameters:
        ...

The workflow in this case will be the following:

* if :math:`\chi^2 < 0.5` the ``Executor`` will continue with the next node
* if :math:`\chi^2 > 0.5` the execution will be interrupted.

Currently the only validator available is the :math:`\chi^2`.

.. note::

  When validation is performed on multiple qubits the execution
  will be stopped if the majority of the qubits fails to satistify
  the validation requirement.


Handling exceptions
^^^^^^^^^^^^^^^^^^^

Qibocal provides a way to deal with validation errors which allows
the user to implement **Exceptional flows** described in :ref:`autoruncard`.

Let's consider a simple example where we run a T1 experiment and if the
:math:`\chi^2` exceeds a specific threshold :math:`\xi_1` we want to run a Rabi
experiment to improve our estimate on T1, while if :math:`\chi^2` exceeds another threshold
:math:`\xi_2 > \xi_1` we decide to stop the execution.


The runcard corresponding to the previous experiment is the following:


.. code-block:: yaml

  qubits: [0]

  actions:
    - id: t1
      priority: 0
      operation: t1
      parameters:
        delay_before_readout_start: 16
        delay_before_readout_end: 2000
        delay_before_readout_step: 20
      validator:
        scheme: chi2
        parameters:
          thresholds: [xi_1, xi_2]
        outcomes: ["rabi"]

    - id: rabi
      priority: 10
      operation: rabi_amplitude
      parameters:
        min_amp_factor: 0.0
        max_amp_factor: 2.0
        step_amp_factor: 0.02
        pulse_length: 40
        relaxation_time: 100_000
        nshots: 1024

The exception is handled using a new entry ``outcomes`` which corresponds to a list
of possible exit conditions linked to the validation outcome. A single outcome could be
either a ``TaskId`` or a tuple including a ``TaskId`` and
a dictionary with modification to the parameters already listed in the node.
If we wish to run a rabi with a pulse length of 30 ns the notation would be the following:

.. code-block:: yaml

  qubits: [0]

  actions:
    - id: t1
      priority: 0
      operation: t1
      parameters:
        delay_before_readout_start: 16
        delay_before_readout_end: 2000
        delay_before_readout_step: 20
      validator:
        scheme: chi2
        parameters:
          thresholds: [0.1, 10]
        outcomes: [["rabi", {"pulse_length" : 30}]]

    - id: rabi
      priority: 10
      operation: rabi_amplitude
      parameters:
        min_amp_factor: 0.0
        max_amp_factor: 2.0
        step_amp_factor: 0.02
        pulse_length: 40
        relaxation_time: 100_000
        nshots: 1024




Qibocal allows the user to put a generic number of thresholds :math:`N` with corresponding
:math:`N-1` outcomes.

.. warning::

  To avoid generating infinite loops there is a variable ``max_iterations`` which can be modified
  by defining globally the varaible ``max_iterations`` in the runcard.
