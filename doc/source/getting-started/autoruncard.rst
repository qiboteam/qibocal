.. _autoruncard:

How to write your automated runcard?
====================================

Automation requires to have a more complex workflow than a simple linear
sequence of steps.
In particular, the following features are requested:

branching
  branch more tasks after a single one

merging
   require multiple routines to have run before

passing data
   values computed from former calibration operations should be available for
   subsequent ones

conditioning
   decide how to proceed based on the calibrations results

In order to implement this features, an adequate representation of the complex
calibration task has to be provided, together with a suitable executor, that
given a task specification and a machine to run on, it is able to perform the
full task in full automation, unless some of the individual operations results
in an unexpected outcome.

To provide a cleaner abstraction over the space of possible complex flows, the
execution has been split in two different regimes, in order to compose the most
general flow of a *simpler* but more common part, and a more complex but fully
generic addition.
These two parts are dubbed:

Normal Flow
   which is defined by a Directed Acyclic Graph (DAG), and

Exceptional Flow
   that allows to branch off at any point from the Normal Flow, and alter it
   according to dynamic conditions

How to use
----------

Only the Normal Flow is currently implemented, and it is defined by a runcard
like the following:

.. code-block:: yaml

   actions:
    - id: start
      priority: 0
      next: [first, third]
      operation:
    - id: first
      priority: 100
      main: second
      next: [fourth]
      operation:
    - id: second
      priority: 150
      operation:
    - id: third
      priority: 300
      next: [fourth]
      operation:
    - id: fourth
      priority: 200
      operation:

   # normal-flow execution: [start, first, second, third, fourth]

.. caution::

   ``operation`` keys are left empty, since no operation is implemented at the
   moment (other than a dummy one, exactly identified by a ``null`` value, that
   is also a default)

.. important::

   Consider dropping the ``main`` key, for the sake of simplicity: it adds
   nothing to priority and the "follow the thread" rule
