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

The execution will be interrupted in this case if the :math:`\chi^2` exceeds
`chi_max_value`.
