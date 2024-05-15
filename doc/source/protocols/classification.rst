Classification experiment
=========================

Experiment Description
----------------------

Calibrates the single shot classification in the IQ plane.

Example Runcard
---------------

.. code-block::

    - id: single shot classification
      operation: single_shot_classification
      parameters:
        nshots: 1024
        classifiers_list: ["qblox_fit", "naive_bayes"]

Parameters
----------

- nshots: number of shots for each state
- classifier_list (optional): type of classifier to test. The available ones are:

  - linear_svm
  - ada_boost
  - gaussian_process
  - naive_bayes
  - qubit_fit
  - random_forest
  - rbf_svm
  - qblox_fit.
