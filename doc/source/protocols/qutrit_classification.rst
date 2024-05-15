Qutrit classification experiment
================================

Experiment Description
----------------------

Calibrates the single shot classification in the IQ plane for the 0-1-2 states.

Example Runcard
---------------

.. code-block::

    - id: single shot classification
      operation: single_shot_classification
      parameters:
        nshots: 1024
        classifiers_list: ["nn", "naive_bayes"]

Parameters
----------

- nshots: number of shots for each state
- classifier_list (optional): type of classifier to test. The available ones are:

  - naive_bayes
  - nn
  - random_forest
  - decision_tree
