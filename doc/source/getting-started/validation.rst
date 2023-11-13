How to add validation to your protocol?
=======================================

In Qibocal there is the possibility to add a validation step
during the execution of your protocols.
Here is an example of a runcard which validates the results through
:math:`\chi^2`.

.. code-block:: yaml

   actions:
    - id: t1
      priority: 0
      operation: t1
      validator:
        scheme: chi2
        parameters:
            chi2_max_value: 1
      parameters:
        ...

The execution will be interrupted in this case if the :math:`\chi^2` exceed
`chi_max_value`.

Adding a custom validator at runtime
------------------------------------

.. TO COMPLETE
The user is also able to define a custom validator through the python API.
For example let's suppose that we want to write a validator that implements
the following condition: "stop if T1 is less than 10us".

.. code-block:: python

   from qibocal.protocols.characterization import Operation
   from qibolab.qubits import QubitId
   from qibocal.auto.status import Normal, Failure
   from qibocal.auto.validators import VALIDATORS

   # retrieve t1 object
   t1 = Operation["t1"].value

   def t1_threshold(results: t1.results_type, qubit: QubitId, t1_min: float = 10e4):
        # store T1 value for qubit
        t1 = results.t1[qubit][0]

        if t1 > t1_min:
            return Normal()
        else:
            return Failure()

    VALIDATORS["t1_threshold"] = t1_threshold
