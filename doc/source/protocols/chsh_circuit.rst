CHSH (circuits)
===============

Experiment Description
----------------------

Perform the CHSH protocol using circuits.

Example Runcard
---------------

.. code-block::

    - id: chsh
      operation: chsh_circuits
      parameters:
        bell_states: [0, 1, 2]
        ntheta: 4
        native: False
        apply_error_mitigation: False

Parameters
----------

- bell_states: list with Bell states to compute CHSH.
    - 0 -> |00>+|11>
    - 1 -> |00>-|11>
    - 2 -> |10>-|01>
    - 3 -> |10>+|01>
- ntheta: number of angles probed linearly between 0 and 2 pi
- native: if True a circuit will be created using only GPI2 and CZ gates
- apply_error_mitigation: error mitigation mode
